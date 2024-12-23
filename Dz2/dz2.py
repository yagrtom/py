# -*- coding: utf-8 -*-

#������������� ��������� �������������� ������� �� ���� �����������


import math

import numpy as np
import numpy.typing as npt

from objects import LayerContinuous, LayerDiscrete, Probe

import boundary
import sources
import tools



class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return round(x / self.discrete)


def sampleLayer(layer_cont: LayerContinuous, sampler: Sampler) -> LayerDiscrete:
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu, layer_cont.sigma)


def fillMedium(layer: LayerDiscrete,
               eps: npt.NDArray[np.float64],
               mu: npt.NDArray[np.float64],
               sigma: npt.NDArray[np.float64]):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
        sigma[layer.xmin: layer.xmax] = layer.sigma
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu
        sigma[layer.xmin:] = layer.sigma


if __name__ == '__main__':

    # ������������ ���������
    # �������� ������������� ���������� ������������
    W0 = 120.0 * np.pi

    # �������� ����� � �������
    c = 299792458.0

    # ������������� ����������
    eps0 = 8.854187817e-12

     # ��������� �������������
    # ������� �������, ��
    f_min = 1e9
    f_max = 4e9
    f_mid = (f_min+f_max)/2

    # ������� �� ������������ � �
    # for test Hz 0.2*1e-4
    # for model 2*1e-4
    dx = 0.5*1e-4
    
    wavelength = c / f_mid
    Nl = wavelength / dx

    # ����� �������
    Sc = 1.0

    # ������ ������� ������������� � �
    maxSize_m = 1.4

    # ����� ������� � ��������
    # for test 1e-9
    # for model 4e-8 
    maxTime_s = 15e-9

    # ��������� ��������� � �
    sourcePos_m = 0.1

    # ��������� �����
    layers_cont = [LayerContinuous(0.5, 0.71,  eps=7.8, sigma=0.0),
                   LayerContinuous(0.71, 1.05 , eps=4.2, sigma=0.0),
                   LayerContinuous(1.05, eps=5.5, sigma=0.0)
                   ]
    # ��������� �����

    # �������� ���������� ������� ����
    speed_refresh = 1000

    # ������� � ���������� ��������
    # ������� �� �������
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # ����� ������� � ��������
    maxTime = sampler_t.sample(maxTime_s)

    # ������ ������� ������������� � ��������
    maxSize = sampler_x.sample(maxSize_m)

    # ��������� ��������� � ��������
    sourcePos = sampler_x.sample(sourcePos_m)

    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # ���������� �������� ��� ����������� ���� � �
    coord_probe1_m = 0.08
    coord_probe2_m = 0.11
    probesPos_m = [coord_probe1_m, coord_probe2_m]
    # ������� ��� ����������� ����
    probesPos = [sampler_x.sample(pos) for pos in probesPos_m]
    probes = [Probe(pos, maxTime) for pos in probesPos]


    # ��� ���������� ����� ��������� ������� ������� �������������� ������
    posSP = 0.11
    signal_probe = Probe(sampler_x.sample(posSP), maxTime)

    # ��������� �����
    # ��������������� �������������
    eps = np.ones(maxSize)

    # ��������� �������������
    mu = np.ones(maxSize - 1)

    # ������������
    sigma = np.zeros(maxSize)

    for layer in layers:
        fillMedium(layer, eps, mu, sigma)

    # ������������ ��� ����� ������


    # ��������
    #magnitude = 3.0
    #signal = sources.HarmonicPlaneWave.make_continuous(magnitude, f_Hz, dt, Sc,
    #                                                   eps[sourcePos],
    #                                                   mu[sourcePos])
    #source = sources.SourceTFSF(signal, 0.0, Sc, eps[sourcePos], mu[sourcePos])
    #source = sources.Harmonic.make_continuous(magnitude, f_Hz, dt, Sc)




    # ��������
    aplitude = 1.0
    source = sources.ModGauss(5000, 2500, Nl, Sc, eps[sourcePos], mu[sourcePos])
    # source = sources.Harmonic.make_continuous(magnitude, f_Hz, dt, Sc)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)

    # �������� ����������� ������� ��������� �������
    boundary_left = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
    boundary_right = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

    # ��������� ����������� ���� E
    display_field = Ez
    display_ylabel = 'Ez, B/m'
    display_ymin = -2.1
    display_ymax = 2.1

    # �������� ���������� ������ ��� �����������
    # ������������� ���� � ������������
    display = tools.AnimateFieldDisplay(dx, dt,
                                        maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel,
                                        title='fdtd_dielectric')

    display.activate()
    display.drawSources([sourcePos])
    display.drawProbes(probesPos)
    for layer in layers:
        display.drawBoundary(layer.xmin)
        if layer.xmax is not None:
            display.drawBoundary(layer.xmax)

    for t in range(1, maxTime):
        # ������ ���������� ���� H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # �������� ����������� � �������������� ������
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, t)
        # ��������� ������� ��� ���� E - PEC

        # ��������� ������� ��� ���� E - PEC

        # ������ ���������� ���� E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]

        # �������� ����������� � �������������� ������
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, t + 0.5))

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        if(t < 10000):
            signal_probe.addData(Ez,Hy)
            
        # for test
        #if(t == 4800):
        #    signal_probe.addData(Ez,Hy)

        # ����������� ���� � ��������
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % speed_refresh == 0:
            display.updateData(display_field, t)

    display.stop()

    # ����������� �������, ������������ � ��������
    
    #tools.showProbeAll(signal_probe, probes, dx, dt, -2.1, 2.1)
    
    tools.showProbeFalRefSignalMy(signal_probe, probes[0], dx, dt, -2.1, 2.1)
    
    tools.showProbeFalRefSpectrumMy(signal_probe, probes[0], dx, dt, -2.1, 2.1)

    tools.showProbeKoeReflectedOfFrequency(signal_probe, probes[0], dx, dt, -2.1, 2.1)
