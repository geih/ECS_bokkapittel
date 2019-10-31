#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import LFPy
import neuron
h = neuron.h
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from plotting_convention import mark_subplots, simplify_axes


np.random.seed(1234)

################################################################################
# Main script, set parameters and create cell, synapse and electrode objects
################################################################################


def insert_synapses(cell, synparams, section, n, spTimesFun, args, z_min=-1e9, z_max=1e9):
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n, z_min=z_min, z_max=z_max)
    # idx = cell.get_rand_idx_area_and_distribution_norm(section=section,
    #                                    funargs=dict(loc=500, scale=150),
    #                                        nidx=n, z_min=z_min, z_max=z_max)
    synapses = []
    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})

        # Some input spike train using the function call
        [spiketimes] = spTimesFun(**args)

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(np.array([spiketimes]))
        synapses.append(s)
    return synapses


def remove_active_mechanisms():
    remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
                   "SK_E2", "K_Tst", "K_Pst", "Im", "Ih",
                   "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove()


def remove_ca_mechanisms():
    remove_list = ["SK_E2", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove()


def make_cell_uniform(Vrest=-70):
    """
    Adjusts e_pas to enforce a uniform resting membrane potential at Vrest
    """
    neuron.h.t = 0
    neuron.h.finitialize(Vrest)
    neuron.h.fcurrent()
    for sec in neuron.h.allsec():
        for seg in sec:
            seg.e_pas = seg.v
            if neuron.h.ismembrane("na_ion"):
                seg.e_pas += seg.ina/seg.g_pas
            if neuron.h.ismembrane("k_ion"):
                seg.e_pas += seg.ik/seg.g_pas
            if neuron.h.ismembrane("ca_ion"):
                seg.e_pas = seg.e_pas + seg.ica/seg.g_pas
            if neuron.h.ismembrane("Ih"):
                seg.e_pas += seg.ihcn_Ih/seg.g_pas


def return_cell(cell_name, sim_name):

    if cell_name == "hay":

        neuron.load_mechanisms(join('L5bPCmodelsEH', "mod"))
        cell_parameters = {
            'morphology': join('L5bPCmodelsEH/morphologies/cell1.asc'),
            'templatefile': [join('L5bPCmodelsEH/models/L5PCbiophys3.hoc'),
                               join('L5bPCmodelsEH/models/L5PCtemplate.hoc')],
            'templatename': 'L5PCtemplate',
            'templateargs': join('L5bPCmodelsEH/morphologies/cell1.asc'),
            'passive': False,
            'nsegs_method': None,
            'dt': 2**-6,
            'tstart': -150,
            'tstop': 10,
            'v_init': -70,
            'celsius': 34,
            'pt3d': True,
        }

        cell = LFPy.TemplateCell(**cell_parameters)

        if "without_ca" in sim_name:
            remove_ca_mechanisms()
            make_cell_uniform(Vrest=-70)
            print("Removed calcium currents")
        elif "passive" in sim_name:
            remove_active_mechanisms()
            make_cell_uniform(Vrest=-70)
            print("Removed all active currents")
        else:
            print("Control simulation")
            make_cell_uniform(Vrest=-70)
        cell.set_rotation(x=4.729, y=-3.166)


    elif cell_name == "two_comp":
        h("""
        proc celldef() {
          topol()
          subsets()
          geom()
          biophys()
          geom_nseg()
        }

        create axon[1]

        proc topol() { local i
          basic_shape()
        }
        proc basic_shape() {
          axon[0] {pt3dclear()
          pt3dadd(0, 0, 0, 5)
          pt3dadd(0, 0, 1400, 5)}
        }

        objref all
        proc subsets() { local i
          objref all
          all = new SectionList()
            axon[0] all.append()

        }
        proc geom() {
        }
        proc geom_nseg() {
            forall {nseg = 2}
        }
        proc biophys() {
        }
        celldef()

        Ra = 100.
        cm = 1.
        Rm = 30000

        forall {
            insert pas
            }
        """)

        dt = 2**-4
        cell_params = {          # various cell parameters,
                    'morphology': h.all,
                    'delete_sections': False,
                    'v_init': -70.,    # initial crossmembrane potential
                    'passive': False,   # switch on passive mechs
                    'nsegs_method': None,
                    'lambda_f': 1000.,
                    'dt': dt,   # [ms] dt's should be in powers of 2 for both,
                    'tstart': 0.,    # start time of simulation, recorders start at t=0
                    'tstop': 10.,
                    "pt3d": True,
                }

        cell = LFPy.Cell(**cell_params)
        # cell.set_pos(x=-cell.xstart[0])

    return cell


def plot_one_LFP_timestep(cell, grid_LFP, laminar_LFP, time_idx, grid_x, grid_z,
                      laminar_elec_params, synapse, synapse_params, sim_name, savefolder):

    lam_elec_clr = lambda idx: plt.cm.Blues(0.1 + idx / (laminar_LFP.shape[0] + 1))
    soma_clr = 'orange'
    tuft_clr = 'cyan'
    input_site_clr = 'r'
    tuft_idx = np.argmax(cell.zend)

    grid_LFP = grid_LFP[:, time_idx].reshape(grid_x.shape)

    plt.close("all")
    fig = plt.figure(figsize=[3, 5])

    # ax_lam = fig.add_axes([0.05, 0.3, 0.35, 0.6], title="Laminar LFP",
    #                       xticks=[],  frameon=False)
    # ax_lam.set_yticks([0, 1000])
    # ax_lam.set_yticklabels(["0 $\mu$m", "1000 $\mu$m"],
    #                        fontsize=7, rotation=90, va="center")

    ax_c = fig.add_axes([.05, .55, .95, .45], xticks=[], yticks=[],
                      aspect=1, frameon=False,)

    ax = fig.add_axes([.05, .05, .95, .45], xticks=[], yticks=[],
                      aspect=1, frameon=False,)
                      #title="LFP at t = {:2.02f} ms".format(cell.tvec[time_idx]))
    cax = fig.add_axes([0.01, 0.1, 0.3, 0.01], frameon=False)

    # ax_vm = fig.add_axes([.32, .10, .12, .15], title=r'Input site V$_m$',
    #                      ylabel='mV', xlabel="Time (ms)")

    # if hasattr(synapse, 'i'):
    #     ax_syn = fig.add_axes([.11, .1, .12, .15], ylabel=r'$i_\mathrm{syn}$ (pA)',
    #                           xlabel="Time (ms)")
    #     ax_syn.axvline(cell.tvec[time_idx], c='gray', ls="--")
    # ax_vm.axvline(cell.tvec[time_idx], c='gray', ls="--")
    # ax_lam.axvline(cell.tvec[time_idx], c='gray', ls="--")

    # zs = laminar_elec_params["z"]
    # dz = np.abs(zs[1] - zs[0])
    max_lfp = np.max(np.abs(laminar_LFP))
    # laminar_normalize = dz / max_lfp
    # for elec_idx in range(laminar_LFP.shape[0]):
    #     zpos = zs[elec_idx]
    #     c = lam_elec_clr(elec_idx)
        # y_sig = laminar_LFP[elec_idx] * laminar_normalize + zpos
        # ax_lam.plot(cell.tvec, y_sig, c=c)
        # ax.plot(laminar_elec_params['x'][elec_idx],
        #         laminar_elec_params['z'][elec_idx], 'o', c=c)

    # ax_lam.plot([cell.tvec[-1] - 10, cell.tvec[-1]], [-100, -100],
    #             lw=2, c='k')
    # ax_lam.text(cell.tvec[-1] - 5, -120, "10 ms", va="top", ha="center")

    # test_magnitudes = 10**np.arange(-2., 4.)
    # print(test_magnitudes - max_lfp)
    # scale_bar_length = test_magnitudes[np.argmin(np.abs(np.log(test_magnitudes/max_lfp)))]
    # scale_bar_length = int(scale_bar_length) if scale_bar_length >= 1.0 else scale_bar_length

    # print(max_lfp, scale_bar_length)
    # ax_lam.plot([cell.tvec[-1], cell.tvec[-1]],
    #             [-100, -100 + scale_bar_length * laminar_normalize],
    #             lw=2, c='k')
    # ax_lam.text(cell.tvec[-1] + 3, -100 + dz/2, "{} nV".format(scale_bar_length),
    #             va="center", ha="left", clip_on=False, zorder=100)

    #plot morphology
    # zips = []
    # for x, z in cell.get_idx_polygons():
    #     zips.append(list(zip(x, z)))
    # polycol = PolyCollection(zips,
    #                          edgecolors='none',
    #                          facecolors='green')
    # ax.add_collection(polycol)
    #

    imem = cell.imem - cell.imem[:, 0, None]

    imem_normalize = 500 / np.max(np.abs(imem[:, time_idx]))
    for idx in range(cell.totnsegs):
        ax.plot([cell.xstart[idx], cell.xend[idx]],
                [cell.zstart[idx], cell.zend[idx]], lw=1, c='gray')

        ax_c.plot([cell.xstart[idx], cell.xend[idx]],
                [cell.zstart[idx], cell.zend[idx]], lw=1, c='gray', zorder=-1)
        i_length = imem[idx, time_idx] * imem_normalize

        if i_length < 0:
            x_start = cell.xmid[idx] - i_length
            c = 'b'
            print(idx, cell.imem[idx, time_idx])
        else:
            x_start = cell.xmid[idx]
            c = 'r'
        # if np.abs(i_length) > 10:
        ax_c.arrow(x_start, cell.zmid[idx], i_length, 0,
                   color=c, clip_on=False, fc=c, head_width=np.abs(i_length/500))
        # ax_c.plot([cell.xmid[idx], cell.xmid[idx] + ],
        #           [cell.zmid[idx], cell.zmid[idx]])

    # ax.plot([200, 300], [-200, -200], 'k', lw=2, clip_on=False)
    # ax.text(250, -250, r'100 $\mu$m', va='center', ha='center')

    #ax.plot(cell.xmid[cell.synidx], cell.zmid[cell.synidx], 'D', ms=5,
    #        mec='k', mfc=input_site_clr)
    #ax.plot(cell.xmid[cell.somaidx[0]], cell.zmid[cell.somaidx[0]], 'D', ms=5,
    #        mec='k', mfc=soma_clr)
    #ax.plot(cell.xmid[tuft_idx], cell.zmid[tuft_idx], 'D', ms=5,
    #        mec='k', mfc=tuft_clr)
    # if hasattr(synapse, 'i'):
    #     ax_syn.plot(cell.tvec, synapse.i*1E3, color=input_site_clr, clip_on=False)
    # if "idx" in synapse_params:
    #     ax_vm.plot(cell.tvec, cell.vmem[synapse_params["idx"], :], c=input_site_clr)
    # ax_vm.plot(cell.tvec, cell.vmem[cell.somaidx[0], :], c=soma_clr)
    # ax_vm.plot(cell.tvec, cell.vmem[tuft_idx, :], c=tuft_clr, ls="--")

    num = 15
    levels = np.logspace(-2, 0, num=num)
    scale_max = 1000.#10**np.ceil(np.log10(np.max(np.abs(LFP)))) / 10
    # print(scale_max, np.max(np.abs(grid_LFP)))
    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    #rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu
    rainbow_cmap = plt.cm.get_cmap('bwr_r') # rainbow, spectral, RdYlBu
    
    colors_from_map = [rainbow_cmap(i*np.int(255/(len(levels_norm) - 2))) for i in range(len(levels_norm) -1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    # ticks = [levels_norm[2*i] for i in range(int(num/2) + 1)] + [levels_norm[num + 2*i] for i in range(int(num/2) + 1)]

    ep_intervals = ax.contourf(grid_x, grid_z, grid_LFP,
                               zorder=-2, colors=colors_from_map,
                               levels=levels_norm, extend='both')

    ax.contour(grid_x, grid_z, grid_LFP, colors='k', linewidths=(1), zorder=-2,
               levels=levels_norm)

    cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal',
                 format='%.0E', extend='max')

    cbar.set_ticks([-1000, -100, 0,  100, 1000])
    cax.set_xticklabels([-1000, -100, 0, 100, 1000],
                        fontsize=8, rotation=45)
    cbar.set_label('$\phi$ (nV)', labelpad=-5)

    # for ax in [ax_vm]:
    #     for loc, spine in ax.spines.items():
    #         if loc in ['right', 'top']:
    #             spine.set_color('none')
    #     ax.xaxis.set_ticks_position('bottom')
    #     ax.yaxis.set_ticks_position('left')

    plt.savefig(join(savefolder, 'lfp_{}_{:04d}.png'.format(sim_name, time_idx)), dpi=300)


def plot_EAP(cell, grid_LFP, laminar_LFP, grid_x, grid_z,
                          laminar_elec_params, sim_name, savefolder):

    lam_elec_clr = ['pink', 'cyan']
    plot_apic = False



    soma_clr = 'orange'
    apic_clr = "olive"
    apic_idx = cell.get_closest_idx(x=0, y=0, z=600)


    plt.close("all")
    fig = plt.figure(figsize=[3, 4])

    ax = fig.add_axes([.01, .12, .6, .98], xticks=[], yticks=[],
                      aspect=1, frameon=False, )
    # title="LFP at t = {:2.02f} ms".format(cell.tvec[time_idx]))
    cax = fig.add_axes([0.05, 0.11, 0.4, 0.01], frameon=False)

    ax_vm = fig.add_axes([.67, .13, .25, .27], title='membrane\npotential', #ylim=[-80, 50],
                         xlabel="Time (ms)")
    ax_lam = fig.add_axes([0.67, 0.57, 0.25, 0.27], title="extracellular\npotentials",) #ylim=[-150, 50])

    ax_vm.set_ylabel('mV', labelpad=-6)
    ax_lam.set_ylabel('$\mu$V', labelpad=-6)

    for elec_idx in range(laminar_LFP.shape[0]):
        c = lam_elec_clr[elec_idx]
        y_sig = laminar_LFP[elec_idx]
        ax_lam.plot(cell.tvec, y_sig, c=c)

        ax.plot(laminar_elec_params['x'][elec_idx],
                laminar_elec_params['z'][elec_idx], 'o', c=c, zorder=100)

    for idx in range(cell.totnsegs):
        ax.plot([cell.xstart[idx], cell.xend[idx]],
                [cell.zstart[idx], cell.zend[idx]], lw=1, c='gray')

    ax.plot([150, 250], [-300, -300], 'k', lw=2, clip_on=False)
    ax.text(200, -350, r'100 $\mu$m', va='center', ha='center')

    ax.plot(cell.xmid[cell.somaidx[0]], cell.zmid[cell.somaidx[0]], 'D', ms=5,
           mec='k', mfc=soma_clr)

    ax_vm.plot(cell.tvec, cell.vmem[cell.somaidx[0], :], c=soma_clr)

    if plot_apic:
        ax.plot(cell.xmid[apic_idx], cell.zmid[apic_idx], 'D', ms=5,
               mec='k', mfc=apic_clr)

        ax_vm.plot(cell.tvec, cell.vmem[apic_idx, :], c=apic_clr)

    time_idx = 0

    l1 = ax_vm.axvline(cell.tvec[time_idx], ls='--', c='gray')
    l2 = ax_lam.axvline(cell.tvec[time_idx], ls='--', c='gray')

    num = 15
    levels = np.logspace(-2.3, 0, num=num)
    scale_max = 10 # 100.

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    # rainbow_cmap = plt.cm.get_cmap('PRGn') # rainbow, spectral, RdYlBu
    rainbow_cmap = plt.cm.get_cmap('bwr_r')  # rainbow, spectral, RdYlBu

    colors_from_map = [rainbow_cmap(i * np.int(255 / (len(levels_norm) - 2))) for i in range(len(levels_norm) - 1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    # ticks = [levels_norm[2*i] for i in range(int(num/2) + 1)] + [levels_norm[num + 2*i] for i in range(int(num/2) + 1)]

    from mpi4py import MPI

    # MPI stuff we're using
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()

    for time_idx in range(len(cell.tvec))[0::10]:
        if not divmod(time_idx, SIZE)[1] == RANK:
            continue
        # time_idx = np.argmax(np.abs(cell.vmem[0, :] - cell.vmem[0, 0])) if time_idx is None else time_idx
        l1.set_xdata(cell.tvec[time_idx])
        l2.set_xdata(cell.tvec[time_idx])

        grid_LFP_ = grid_LFP[:, time_idx].reshape(grid_x.shape)

        if divmod(time_idx, SIZE)[1] == RANK:
            print(time_idx)

        ep_intervals = ax.contourf(grid_x, grid_z, grid_LFP_,
                                   zorder=-2, colors=colors_from_map,
                                   levels=levels_norm, extend='both')

        ax.contour(grid_x, grid_z, grid_LFP_, colors='k', linewidths=(1), zorder=-2,
                   levels=levels_norm)

        cbar = fig.colorbar(ep_intervals, cax=cax, orientation='horizontal',
                            format='%.0E', extend='max')

        cbar.set_ticks(np.array([-1, -0.1,  0,  0.1, 1]) * scale_max)
        cax.set_xticklabels(np.array(np.array([-1, -0.1, 0, 0.1, 1]) * scale_max, dtype=int),
                            fontsize=7, rotation=45)
        cbar.set_label('$\phi$ ($\mu$V)', labelpad=-5)

        simplify_axes([ax_vm, ax_lam])

        plt.savefig(join(savefolder, 'eap_{}_{:04d}.png'.format(sim_name, time_idx)), dpi=150)


def plot_spatial_time_traces(cell, grid_LFP, grid_elec_params,
                          sim_name, savefolder):

    plt.close("all")
    fig = plt.figure(figsize=[5, 4])

    ax = fig.add_axes([.01, .1, .6, .85], xticks=[], yticks=[],
                      aspect=1, frameon=False, xlim=[-150, 150], ylim=[-100, 300])

    ax_vm = fig.add_axes([.75, .2, .2, .6], title='somatic\nmembrane\npotential', #ylim=[-80, 50],
                         xlabel="Time (ms)")

    ax_vm.set_ylabel('mV', labelpad=-1)

    dx = np.diff(sorted(set(grid_electrode.x)))[0]
    dz = grid_electrode.z[1] - grid_electrode.z[0]
    sig_shrink_factor = 0.7

    ep_sig_scale = 100

    norm_LFP = dz / ep_sig_scale

    print(np.max(np.abs(grid_LFP)))

    cell_clr = "0.8"

    #plot morphology
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips,
                             edgecolors=cell_clr, lw=0.5,# alpha=0.2,
                             facecolors=cell_clr)
    ax.add_collection(polycol)
    ax.plot(cell.somapos[0], cell.somapos[1], 'o', c=cell_clr, ms=30)

    for elec_idx in range(grid_LFP.shape[0]):
        x = grid_elec_params['x'][elec_idx]
        z = grid_elec_params['z'][elec_idx]

        t_ = cell.tvec / cell.tvec[-1] * dx * sig_shrink_factor + x
        y_sig = grid_LFP[elec_idx] * norm_LFP * sig_shrink_factor + z

        ax.plot(t_, y_sig, c='k', clip_on=False)
        ax.plot(x, z, '.', c='k', zorder=100, clip_on=False)

    ax.plot([25, 75], [-110, -110], cell_clr, lw=2, clip_on=False)
    ax.text(50, -120, r'50 $\mu$m', va='center', ha='center', color=cell_clr)

    ax.plot([-75, -75 + dx * sig_shrink_factor], [-110, -110], 'k', lw=2, clip_on=False)
    ax.text(-95, -120, r'10 ms', va='center', ha='left', color='k')

    ax.plot([-75 + dx * sig_shrink_factor, -75 + dx * sig_shrink_factor],
            [-110, -110- dz * sig_shrink_factor], 'k', lw=2, clip_on=False)
    ax.text(-35, -130, r'{:d} $\mu$V'.format(ep_sig_scale), va='center', ha='left', color='k')

    ax_vm.plot(cell.tvec, cell.vmem[cell.somaidx[0], :], c='k')
    simplify_axes([ax_vm])

    plt.savefig(join(savefolder, 'traces_{}.png'.format(sim_name)), dpi=150)


synapse_type = ["single_tuft", "single_soma"][1]
cell_name = ["two_comp", "hay"][1]

sim_name = '{}_{}_{}_traces'.format(cell_name, synapse_type, "spike")
savefolder = join(sim_name)

os.makedirs(savefolder, exist_ok=True)

cell = return_cell(cell_name, sim_name)

# Define synapse parameters
synapse_params = {
    'e' : 0.,                   # reversal potential
    'syntype' : 'Exp2Syn',       # synapse type
    'tau1' : 0.1,                 # synaptic time constant
    'tau2' : 1.,                 # synaptic time constant
    'weight' : 0.001,            # synaptic weight
    'record_current' : True,    # record synapse current
}

if "soma" in synapse_type:
    synapse_params["idx"] = cell.get_closest_idx(x=0, z=0)
elif "tuft" in synapse_type:
    synapse_params["idx"] = cell.get_closest_idx(x=100, z=1400)

synapse_params["weight"] = 0.05#0.2
synapses = [LFPy.Synapse(cell, **synapse_params)]
# synapses[0].set_spike_times(np.array([1., 5., 9.]))
synapses[0].set_spike_times(np.array([1.]))

# Create a grid of measurement locations, in (mum)
# grid_x, grid_z = np.mgrid[-550:551:25, -600:1501:25]
# grid_x, grid_z = np.mgrid[-75:76:50, -75:76:50]
grid_x, grid_z = np.mgrid[-75:76:50, -75:251:50]
grid_y = np.zeros(grid_x.shape)

# Define electrode parameters
grid_elec_params = {
    'sigma': 0.3,      # extracellular conductivity
    'x': grid_x.flatten(),  # electrode requires 1d vector of positions
    'y': grid_y.flatten(),
    'z': grid_z.flatten()
}

# Create a grid of measurement locations, in (mum)
laminar_z = np.array([0, 200])
laminar_x = np.ones(len(laminar_z)) * 30
laminar_y = np.zeros(len(laminar_z))
# Define electrode parameters
laminar_elec_params = {
    'sigma': 0.3,      # extracellular conductivity
    'x': laminar_x,  # electrode requires 1d vector of positions
    'y': laminar_y,
    'z': laminar_z
}

# Run simulation, electrode object argument in cell.simulate
print("running simulation...")
cell.simulate(rec_imem=True, rec_vmem=True)

laminar_electrode = LFPy.RecExtElectrode(cell, **laminar_elec_params)
laminar_electrode.calc_lfp()
laminar_LFP = 1e3 * laminar_electrode.LFP

grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
grid_electrode.calc_lfp()
grid_LFP = 1e3 * grid_electrode.LFP

grid_LFP -= grid_LFP[:, 0, None]

max_elec, max_time_idx = np.unravel_index(np.abs(laminar_LFP).argmax(), grid_LFP.shape)

# plot_EAP(cell, grid_LFP, laminar_LFP, None,
#                             grid_x, grid_z, laminar_elec_params,  sim_name, savefolder)

plot_spatial_time_traces(cell, grid_LFP, grid_elec_params,  sim_name, savefolder)

# plot_EAP(cell, grid_LFP, laminar_LFP,
#                  grid_x, grid_z, laminar_elec_params, sim_name, savefolder)
# #     print(time_idx)
#     plot_one_LFP_timestep(cell, grid_LFP, laminar_LFP, time_idx,
#                            grid_x, grid_z, laminar_elec_params, synapses,
#                            synapse_params, sim_name, savefolder)
#     sys.exit()