#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from os.path import join
import numpy as np
import matplotlib
# matplotlib.use("AGG")
import LFPy
import neuron
h = neuron.h
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# from npy_invisible import neuron_models
# from npy_invisible import plot_lfp

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
                                    "CaDynamics_E2", "Ca_LVAst", "Ca"]
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
            if neuron.h.ismembrane("Ih_frozen"):
                seg.e_pas += seg.ihcn_Ih_frozen/seg.g_pas

def return_cell(cell_name):

    if cell_name == "hay":
        neuron.load_mechanisms(join('L5bPCmodelsEH', "mod"))
        ##define cell parameters used as input to cell-class
        cell_parameters = {
            'morphology'    : join('L5bPCmodelsEH/morphologies/cell1.asc'),
            'templatefile'  : [join('L5bPCmodelsEH/models/L5PCbiophys3.hoc'),
                               join('L5bPCmodelsEH/models/L5PCtemplate.hoc')],
            'templatename'  : 'L5PCtemplate',
            'templateargs'  : join('L5bPCmodelsEH/morphologies/cell1.asc'),
            'passive' : False,
            'nsegs_method' : None,
            'dt' : 2**-4,
            'tstart' : -500,
            'tstop' : 10,
            'v_init' : -70,
            'celsius': 34,
            'pt3d' : True,
        }

        cell = LFPy.TemplateCell(**cell_parameters)
        remove_active_mechanisms()
        cell.set_rotation(x=4.729, y=-3.166)
        make_cell_uniform(Vrest=-70)


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
                    'v_init' : -70.,    # initial crossmembrane potential
                    'passive' : False,   # switch on passive mechs
                    'nsegs_method' : None,
                    'lambda_f' : 1000.,
                    'dt' : dt,   # [ms] dt's should be in powers of 2 for both,
                    'tstart' : 0.,    # start time of simulation, recorders start at t=0
                    'tstop' : 10.,   # stop simulation at 200 ms. These can be overridden
                                        # by setting these arguments i cell.simulation()
                    "pt3d": True,
                }

        cell = LFPy.Cell(**cell_params)
        # cell.set_pos(x=-cell.xstart[0])

    return cell


def plot_one_timestep(cell, grid_LFP, laminar_LFP, time_idx, grid_x, grid_z,
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


synapse_type = ["single_tuft", "single_soma"][0]
cell_name = ["two_comp", "hay"][0]

sim_name = '{}_{}'.format(cell_name, synapse_type)
savefolder = join(cell_name, sim_name)

if not os.path.isdir(savefolder):
    os.makedirs(savefolder)

# depth = 40
# depth = 165
depth = 600


cell = return_cell(cell_name)


# Define synapse parameters
synapse_params = {
    'e' : 0.,                   # reversal potential
    'syntype' : 'Exp2Syn',       # synapse type
    'tau1' : 0.1,                 # synaptic time constant
    'tau2' : 2.,                 # synaptic time constant
    'weight' : 0.001,            # synaptic weight
    'record_current' : True,    # record synapse current
}

insert_synapses_GABA_A_args = {
    'section' : 'allsec',
    'z_min': np.max(cell.zend) - depth,
    'n' : 100,
    'spTimesFun' : np.random.normal,#LFPy.inputgenerators.get_activation_times_from_distribution,
    'args' : dict(size=1, #tstart=50, tstop=60,
                  # distribution=scipy.stats.norm,
                  # rvs_args=dict(loc=50, scale=0.001)
                  loc=20,
                  scale=2,
                  )
}

if "single" in synapse_type:
    if "soma" in synapse_type:
        synapse_params["idx"] = cell.get_closest_idx(x=0, z=0)
    elif "tuft" in synapse_type:
        synapse_params["idx"] = cell.get_closest_idx(x=100, z=1400)

    synapse_params["weight"] = 0.1
    synapses = [LFPy.Synapse(cell, **synapse_params)]
    synapses[0].set_spike_times(np.array([1.]))
else:
    synapses = insert_synapses(cell, synapse_params, **insert_synapses_GABA_A_args)

# Create a grid of measurement locations, in (mum)
grid_x, grid_z = np.mgrid[-700:701:25, -700:1701:25]
grid_y = np.zeros(grid_x.shape)

# Define electrode parameters
grid_elec_params = {
    'sigma': 0.3,      # extracellular conductivity
    'x': grid_x.flatten(),  # electrode requires 1d vector of positions
    'y': grid_y.flatten(),
    'z': grid_z.flatten()
}


# Create a grid of measurement locations, in (mum)
laminar_z = np.linspace(-200, 800, 11)
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

# [plt.plot(cell.tvec, cell.imem[idx, :]) for idx in range(cell.totnsegs)]
# plt.savefig("test.png")
# plt.show()

laminar_electrode = LFPy.RecExtElectrode(cell, **laminar_elec_params)
laminar_electrode.calc_lfp()
laminar_LFP = 1e6 * laminar_electrode.LFP

grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)
grid_electrode.calc_lfp()
grid_LFP = 1e6 * grid_electrode.LFP

grid_LFP -= grid_LFP[:, 0, None]

max_elec, max_time_idx = np.unravel_index(np.abs(laminar_LFP).argmax(), grid_LFP.shape)

for time_idx in range(len(cell.tvec))[0::20]:
#     print(time_idx)
    plot_one_timestep(cell, grid_LFP, laminar_LFP, time_idx,
                           grid_x, grid_z, laminar_elec_params, synapses,
                           synapse_params, sim_name, savefolder)
