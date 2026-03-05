from pathlib import Path
from typing import OrderedDict, Union
from typing_extensions import Self

import pickle as pkl
import numpy as np
import pandas as pd
import sif_parser as sif_reader
import scipy.interpolate
import scipy.ndimage
import scipy.signal


def robust_median(mat, axis=None) -> np.ndarray:
    m = 3.0

    d = mat - np.median(mat, axis=axis, keepdims=True)
    mdev = np.median(np.abs(d), axis=axis, keepdims=True)
    s = np.abs(d) / mdev

    newmat = np.copy(mat)
    newmat[s > m] = np.nan

    frame = np.nanmedian(newmat, axis=axis)

    return frame

def wavelength_to_wavenumber(lambda_):
    return 1 / lambda_


def wavenumber_to_wavelength(nu):
    return 1 / nu


class Spectrum:
    data: np.ndarray
    info: OrderedDict
    lambda_: Union[np.ndarray, None]

    _axis_lambda = 0

    def __init__(self, data, info=None, lambda_=None, lax=None):
        self.data = data
        self.info = info  # type: ignore
        self.lambda_ = lambda_  # type: ignore

        lax = 0 if lax is None else lax
        self._axis_lambda = lax

    def __getitem__(self, key):
        key = tuple((x,) if type(x) is int else x for x in key)
        new = self.copy()
        new.lambda_ = (
            (new.lambda_[key[self._axis_lambda]] if new.lambda_ is not None else None)
            if self._axis_lambda is not None
            else None
        )
        new.data = new.data[key]
        return new

    def __sub__(self, other: Union[np.ndarray, Self]):
        if isinstance(other, np.ndarray):
            return Spectrum(
                self.data - other, self.info, self.lambda_, self._axis_lambda
            )

        _check_wavelength_axis(self.lambda_, other.lambda_)
        return Spectrum(
            self.data - other.data, self.info, self.lambda_, self._axis_lambda
        )

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            return Spectrum(
                self.data + other, self.info, self.lambda_, self._axis_lambda
            )

        _check_wavelength_axis(self.lambda_, other.lambda_)
        return Spectrum(
            self.data + other.data, self.info, self.lambda_, self._axis_lambda
        )

    def __truediv__(self, other: Self) -> Self:
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, float)
            or isinstance(other, int)
        ):
            return Spectrum(
                self.data / other, self.info, self.lambda_, self._axis_lambda
            )

        _check_wavelength_axis(self.lambda_, other.lambda_)
        return Spectrum(
            self.data / other.data, self.info, self.lambda_, self._axis_lambda
        )

    def __mul__(self, other: Union[Self, int, float, np.ndarray]) -> Self:
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, float)
            or isinstance(other, int)
        ):
            return Spectrum(
                self.data * other, self.info, self.lambda_, self._axis_lambda
            )

        _check_wavelength_axis(self.lambda_, other.lambda_)
        return Spectrum(
            self.data * other.data, self.info, self.lambda_, self._axis_lambda
        )

    def __matmul__(self, other: Union[Self, np.ndarray]):
        if isinstance(other, np.ndarray):
            return self.interp(other, inplace=False)
        else:
            return self.interp(other.lambda_, inplace=False)

    def flatten(self, *args, inplace=True, **kwargs):
        if inplace:
            self.data = self.data.flatten(*args, **kwargs)
            self.lambda_ = self.lambda_.flatten()
            return self
        else:
            return self.data.flatten(*args, **kwargs)

    def squeeze(self, *args, inplace=True, **kwargs) -> Union[np.ndarray, Self]:
        if inplace:
            self.data = np.squeeze(self.data, *args, **kwargs)
            return self
        else:
            return np.squeeze(self.data, *args, **kwargs)

    @property
    def c(self) -> Self:
        return self.copy()

    @property
    def sdata(self) -> np.ndarray:
        return self.squeeze(inplace=False)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def lambdanm(self) -> np.ndarray:
        return self.lambda_ * 1e9

    @property
    def dnucm(self) -> np.ndarray:
        return self.dnu() / 1e2

    def sort(self):
        ids = np.argsort(self.lambda_)
        self.lambda_ = self.lambda_[ids]
        if self._axis_lambda == -1:
            self.data = self.data[..., ids]
        elif self._axis_lambda == 0:
            self.data = self.data[ids, ...]

        return self

    def bin(
        self, N_bin=8, axis: Union[tuple[int], int] = 0, method="mean", inplace=True
    ):
        if type(axis) is tuple:
            s = self
            for i in axis:
                s.bin(N_bin, axis=i, method=method, inplace=True)
            return s

        if axis == self._axis_lambda:
            newlambda = self.lambda_.reshape(
                *self.lambda_.shape[0:axis], -1, N_bin, *self.lambda_.shape[axis + 1 :]
            )
            # newlambda = self.lambda_.reshape(
            #     -1,
            #     N_bin,
            #     *self.lambda_.shape[1:],
            # )
        else:
            newlambda = self.lambda_

        newdata = self.data.reshape(
            *self.data.shape[0:axis], -1, N_bin, *self.data.shape[axis + 1 :]
        )

        if method == "mean":
            if axis == self._axis_lambda:
                newlambda = newlambda.mean(axis=axis + 1)
            newdata = newdata.mean(axis=axis + 1)
        elif method == "median":
            if axis == self._axis_lambda:
                newlambda = newlambda.mean(axis=axis + 1)
            newdata = np.median(newdata, axis=axis + 1)
        # elif method == 'middle':
        #     newlambda = newlambda.mean(axis=-1)
        #     newdata = np.median(newdata, axis=-1)

        if inplace:
            self.lambda_ = newlambda
            self.data = newdata
            return self
        else:
            return newdata

    def bin2(
        self,
        N_bin: Union[int, tuple[int, ...]],
        axis: Union[int, tuple[int, ...], None] = None,
        method="median",
        combine_axis=None,
        inplace=True,
    ) -> Self:
        if type(N_bin) is int:
            if axis is None:
                bla = list(range(len(self.data.shape)))
                bla.pop(self._axis_lambda)
                axis = tuple(bla)
            if type(axis) != tuple:
                axis = (int(axis),)  # type: ignore
            binshape = tuple(
                N_bin if i in axis else 1 for i in range(len(self.data.shape))
            )
        else:
            binshape = N_bin

        assert type(binshape) == tuple

        mat = np.copy(self.data)
        origshape = mat.shape

        # Pad binshape with ones to match the number of dimensions of mat
        if len(binshape) != len(origshape):
            binshape = (*binshape, *[1 for _ in range(len(origshape) - len(binshape))])

        endshape = [origshape[i] // binshape[i] for i in range(len(origshape))]

        # Reshape mat to have a "binned" dimension for every original dimension
        newshape = [val for pair in zip(endshape, binshape) for val in pair]
        newdata = mat.reshape(*newshape)

        # Move the binned dimensions to the end
        id_bins = tuple(i * 2 + 1 for i in range(len(origshape)))
        id_bins_new = tuple(-i - 1 for i in range(len(origshape)))
        newdata = np.moveaxis(newdata, id_bins, id_bins_new)

        # Combine the binned dimensions
        newdata = newdata.reshape(*endshape, -1)

        if combine_axis is not None:
            # If frameaxis is negative, count from the end
            if combine_axis < 0:
                combine_axis += len(origshape)

            # Combine frameaxis with the binned dimensions
            newdata = np.moveaxis(newdata, combine_axis, -1)
            newdata = newdata.reshape(
                *tuple(x for i, x in enumerate(endshape) if i != combine_axis), -1
            )

        newlambda = self.lambda_
        if newlambda is not None:
            newlambda = newlambda.reshape(-1, binshape[self._axis_lambda])
            newlambda = newlambda.mean(axis=-1)

        if method == "mean":
            newdata = newdata.mean(axis=-1)
        elif method == "median":
            newdata = np.median(newdata, axis=-1)
        elif method == "robust_median":
            newdata = robust_median(newdata, axis=-1)

        if inplace:
            self.lambda_ = newlambda
            self.data = newdata
            return self
        else:
            return newdata

    def filter(self, size=5, method="median", axes=(0, 1), inplace=True):
        if method == "median":
            data = scipy.ndimage.median_filter(
                self.data, size=size, axes=axes, mode="nearest"
            )
        elif method == "mean":
            data = scipy.ndimage.uniform_filter(
                self.data, size=size, axes=axes, mode="nearest"
            )
        elif method == "bin":
            for axis in axes:
                self.bin(size, axis=axis)
            return self
        elif method == "fft":
            # Compute the 2D Fourier Transform of the image
            f_transform = np.fft.fft2(self.data, axes=axes)

            # Shift the zero frequency component to the center
            f_transform_shifted = np.fft.fftshift(f_transform)

            # Define the dimensions of the Fourier spectrum
            rows, cols, frames = self.data.shape

            # Create a lowpass filter (e.g., a circular mask)
            cutoff_frequency = (
                size  # Adjust this value to control the amount of filtering
            )
            y, x = np.ogrid[-rows / 2 : rows / 2, -cols / 2 : cols / 2]
            lowpass_filter = np.zeros((rows, cols, frames))
            lowpass_filter[x**2 + y**2 <= cutoff_frequency**2, :] = 1

            # Apply the filter to the Fourier spectrum
            filtered_spectrum = f_transform_shifted * lowpass_filter

            # Shift the zero frequency component back to the corner
            filtered_spectrum_shifted = np.fft.fftshift(filtered_spectrum)

            # Compute the inverse Fourier Transform to obtain the filtered image
            data = np.abs(np.fft.ifft2(filtered_spectrum_shifted, axes=axes))
        else:
            raise Exception("Method must be one of [median, mean]")

        if inplace:
            self.data = data
            return self
        else:
            return data

    def median(self, *args, inplace=True, **kwargs):
        x = np.nanmedian(self.data, *args, keepdims=True, **kwargs)
        if inplace:
            self.data = x
            return self
        else:
            return x

    def mean(self, *args, inplace=True, **kwargs):
        # TODO: nanmean or mean?
        x = np.nanmean(self.data, *args, **{"keepdims": True, **kwargs})
        if inplace:
            self.data = x
            return self
        else:
            return x

    def max(self, *args, inplace=True, **kwargs):
        x = np.nanmax(self.data, *args, **{"keepdims": True, **kwargs})
        if inplace:
            self.data = x
            return self
        else:
            return x

    def min(self, *args, inplace=True, **kwargs):
        x = np.nanmin(self.data, *args, **{"keepdims": True, **kwargs})
        if inplace:
            self.data = x
            return self
        else:
            return x

    def suml(self, *args, inplace=True, **kwargs):
        return self.sum(*args, inplace=inplace, **{"axis": self._axis_lambda, **kwargs})

    def sum(self, *args, inplace=True, **kwargs):
        # TODO: nansum or sum?
        if inplace:
            self.data = np.nansum(self.data, *args, **{"keepdims": True, **kwargs})
            return self
        else:
            return np.nansum(self.data, *args, **{"keepdims": True, **kwargs})

    def interp(self, lambda_new, *args, inplace=True, **kwargs):
        data_new = scipy.interpolate.interp1d(
            self.lambda_,
            self.data,
            bounds_error=False,
            axis=self._axis_lambda,
            **kwargs,
        )(lambda_new)
        if inplace:
            self.data = data_new
            self.lambda_ = lambda_new
            return self
        else:
            return Spectrum(data_new, self.info, lambda_new, self._axis_lambda)

    def slice(
        self,
        lambda_start=None,
        lambda_end=None,
        dnu_start=None,
        dnu_end=None,
        inplace=True,
    ):
        idl = self.lambda_ != np.inf
        if not lambda_start is None:
            idl = idl & (self.lambda_ >= lambda_start)
        if not lambda_end is None:
            idl = idl & (self.lambda_ <= lambda_end)
        if not dnu_start is None:
            idl = idl & (self.dnu() >= dnu_start)
        if not dnu_end is None:
            idl = idl & (self.dnu() <= dnu_end)

        # mask_shape = np.broadcast_shapes(self.data.shape, self.lambda_.shape)
        # mask = np.broadcast_to(idl, mask_shape)
        if inplace:
            self.data = self.data[idl]
            self.lambda_ = self.lambda_[idl]
            return self
        else:
            return Spectrum(
                self.data[idl], self.info, self.lambda_[idl], self._axis_lambda
            )

    def shift(self, dlambda=None, dnu=None):
        if dlambda is not None:
            self.lambda_ = self.lambda_ + dlambda
        elif dnu is not None:
            self.lambda_ = wavenumber_to_wavelength(self.nu + dnu)
        return self

    def shiftmax(self, to_lambda=None, to_nu=None):
        imax = np.argmax(np.max(self.data, axis=self._axis_lambda))
        if to_lambda is not None:
            lmax = self.lambda_[imax]
            self.lambda_ = self.lambda_ - lmax + to_lambda
        elif to_nu is not None:
            nmax = self.nu[imax]
            self.nu = self.nu - nmax + to_nu
        return self

    def nan(
        self,
        lambda_start=None,
        lambda_end=None,
        dnu_start=None,
        dnu_end=None,
        inplace=True,
    ):
        idl = self.lambda_ > 0
        if lambda_start is not None:
            idl = idl & (self.lambda_ >= lambda_start)
        if lambda_end is not None:
            idl = idl & (self.lambda_ <= lambda_end)
        if dnu_start is not None:
            idl = idl & (self.dnu() >= dnu_start)
        if dnu_end is not None:
            idl = idl & (self.dnu() <= dnu_end)

        if inplace:
            self.data[idl] = np.nan
            return self
        else:
            data = self.data
            data[idl] = np.nan
            return Spectrum(data, self.info, self.lambda_[idl], self._axis_lambda)

    def normalize(self, inplace=True, axis=2):
        if inplace:
            self.data = self.data / np.nanmax(self.data, axis=axis, keepdims=True)
            return self
        else:
            return Spectrum(
                self.data / np.nanmax(self.data, axis=axis, keepdims=True),
                self.info,
                self.lambda_,
                self._axis_lambda,
            )

    @property
    def nu(self):
        return wavelength_to_wavenumber(self.lambda_)

    @nu.setter
    def nu(self, nu_new):
        self.lambda_ = wavenumber_to_wavelength(nu_new)

    def dnu(self, lambda_laser=532e-9):
        return -(self.nu - wavelength_to_wavenumber(lambda_laser))

    # Utility functions
    def smooth(self, window=5, method="movmean", inplace=True, **kwargs):
        if method == "movmean":
            datap = scipy.ndimage.uniform_filter1d(
                self.data, window, mode="nearest", **kwargs
            )
        elif method == "savgol":
            datap = scipy.signal.savgol_filter(
                self.data, window, polyorder=3, mode="nearest"
            )
        if inplace:
            self.data = datap
            return self
        else:
            return Spectrum(datap, self.info, self.lambda_, self._axis_lambda)

    def copy(self) -> Self:
        return Spectrum(
            np.copy(self.data),
            self.info,
            np.copy(self.lambda_) if self.lambda_ is not None else None,
            lax=self._axis_lambda,
        )

    def print_info(self):
        print(f"Signal floor: {np.min(self.data):.0f}")
        print(f"Signal max:   {np.max(self.data):.0f}")
        for k, v in self.info.items():
            print(f"{k}: {v}")

    @classmethod
    def from_file(cls, sif_file, *args) -> Self:
        if Path(sif_file).suffix == ".sif":
            return cls.from_sif(sif_file, *args)

    @classmethod
    def from_sif(
        cls,
        sif_file,
        nm_range: tuple[int, int] = None,
        use_wavelength_calibration=False,
        new_axes=True,
    ) -> Self:
        data, info = sif_reader.np_open(sif_file)
        if new_axes:
            data = data.T
        if use_wavelength_calibration or (nm_range is None):
            bin_num = info["size"][0]
            bin_lambda = info["xbin"]
            # Nominal pixel amount is 1024.
            px = np.arange(0, bin_num) * bin_lambda + bin_lambda / 2
            lambda_ = (
                np.sum(
                    [(px**i) * info["Calibration_data"][i] for i in range(4)], axis=0
                )
                * 1e-9
            )
        else:
            bin_num = info["size"][0]
            bin_lambda = info["xbin"]
            lambda_ = np.linspace(nm_range[0], nm_range[1], 1024) * 1e-9
            lambda_ = lambda_[int(bin_lambda / 2) :: bin_lambda]
        s = cls(data, info, lambda_, lax=(-1 if new_axes else 0))
        return s

    @classmethod
    def from_csv(cls, csv_file, wavelength_factor: float = 1, header="infer") -> Self:
        df = pd.read_csv(csv_file, header=header)
        return cls(
            df.iloc[:, 1].to_numpy(), None, df.iloc[:, 0].to_numpy() * wavelength_factor
        )

    def to_dict(self):
        return {
            "data": self.data,
            "info": self.info,
            "lambda": self.lambda_,
        }

    def save(self, path, name):
        filename = Path(path).resolve() / (name + ".csv")
        df = pd.DataFrame(columns=["Wavelength_m", "Intensity_arb"])
        df.iloc[:, 0] = self.lambda_
        df.iloc[:, 1] = self.data
        df.to_csv(filename, index=False)
        return filename

    def save_pkl(self, path, name):
        filename = Path(path).resolve() / (name + ".pkl")
        with open(filename, "wb") as f:
            pkl.dump(
                {
                    "obj": self,
                    "lambda": self.lambda_,
                    "data": self.data,
                    "info": self.info,
                },
                f,
            )
        return filename


def _check_wavelength_axis(l1, l2):
    if l1 is None and l2 is None:
        return
    else:
        assert np.allclose(
            l1, l2, atol=5e-11, equal_nan=True
        ), f"Spectra have different wavelength axes: {l1} and {l2}"


if __name__ == "__main__":
    Spectrum.from_file(
        "/Volumes/share/Martijn/230502_Thomson_att1/01_800W_175mbar_20slm_5Wlaser.sif"
    )