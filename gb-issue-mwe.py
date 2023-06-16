import hoomd
import gsd.hoomd
import math
import numpy as np
import matplotlib.pyplot as plt

# minimal system for GB
s = gsd.hoomd.Snapshot()
s.particles.N = 2
s.particles.types = ['A']
s.particles.typeid = [0,0]
s.particles.position = [[1,1,1], [2,1,2]]
s.particles.orientation = [[1, 0, 0, 0], [0, math.cos((math.pi)/4), 0.0, math.cos((math.pi)/4)]]
s.configuration.box = [8, 8, 8, 0, 0, 0]
s.particles.mass = [2] * 2

# arbitrary moments of inertia
mass = 1
I = np.zeros(shape=(3, 3))
for r in s.particles.position:
    I += mass * (np.dot(r, r) * np.identity(3) - np.outer(r, r))
s.particles.moment_inertia = [I[0, 0], I[1, 1], I[2, 2]] * s.particles.N

rigid = hoomd.md.constrain.Rigid()
rigid.body['dimer'] = {
    "constituent_types": ['A', 'A'],
    "positions": [[1,1,1], [2,1,2]],
    "orientations": [[1, 0, 0, 0], [0, math.cos((math.pi)/4), 0.0, math.cos((math.pi)/4)]],
    "charges": [0.0, 0.0],
    "diameters": [1.0, 1.0]
}


# sim setup
cpu = hoomd.device.CPU()
sim = hoomd.Simulation(device=cpu, seed=1)
sim.create_state_from_snapshot(s)

# GB params
lperp = 0.3
lpar = 1.0
sigmin = 2 * min(lperp, lpar)
sigmax = 2 * max(lperp, lpar)

cell = hoomd.md.nlist.Cell(buffer=0.4)
gay_berne = hoomd.md.pair.aniso.GayBerne(nlist=cell)
gay_berne.params[('A', 'A')] = dict(epsilon=1.0, lperp=lperp, lpar=lpar)

# this r_cut param is our focus:
gay_berne.r_cut[('A', 'A')] = 2.5

integrator = hoomd.md.Integrator(dt=0.005, integrate_rotational_dof=True)
rigid_centers_and_free = hoomd.filter.Rigid(("center", "free"))
nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
integrator.methods.append(nve)
integrator.forces.append(gay_berne)
sim.operations.integrator = integrator


# track energies for plotting
logger = hoomd.logging.Logger(categories=['scalar'])
logger.add(gay_berne, quantities=['energy'])

# write energies to plot
Table = hoomd.write.Table(output=open('gay_berne_log.txt', mode='w'),
                          trigger = hoomd.trigger.Periodic(1),
                          logger=logger)
sim.operations.writers.append(Table)

thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
sim.operations.computes.append(thermodynamic_properties)

gsd_writer = hoomd.write.GSD(filename='gb_traj.gsd',
                             trigger=hoomd.trigger.Periodic(1),
                             mode='wb',
                             log=logger,
                             filter=hoomd.filter.All())
sim.operations.writers.append(gsd_writer)

sim.run(1000)

f = gsd.hoomd.open('gb_traj.gsd', 'rb')

def distconvert(dist, lperp, lpar, ei, ej):
    sigmin = min(lperp, lpar) * 2
    H = 2 * (lperp**2) * np.identity(3) + ((lpar**2 - lperp**2) * (np.outer(ei, ei) + np.outer(ej, ej)))
    sigma = (np.dot((0.5*dist) , np.dot(np.linalg.inv(H), dist)))**(-2)
    gbdist = ((np.linalg.norm(dist) - sigma + sigmin) / sigmin)
    return np.abs(gbdist)

def phiconvert(dist, lperp, lpar, ei, ej):
    sigmin = min(lperp, lpar) * 2
    H = 2 * (lperp**2) * np.identity(3) + ((lpar**2 - lperp**2) * (np.outer(ei, ei) + np.outer(ej, ej)))
    sigma = (np.dot((0.5*dist) , np.dot(np.linalg.inv(H), dist)))**(-2)
    gbdist = np.dot((0.5*dist) , np.dot(np.linalg.inv(H), dist))
    return np.abs(gbdist)

distances = []
gbdists = []
phidists = []
energies = []
for frame in f:
    #Adjust for periodic boundaries
    #distvect = frame.particles.position[1] - frame.particles.position[0]
    dx, dy, dz = frame.particles.position[1] - frame.particles.position[0]
    dx = np.abs(dx) % (frame.configuration.box[0]/2)
    dy = np.abs(dy) % (frame.configuration.box[1]/2)
    dz = np.abs(dz) % (frame.configuration.box[2]/2)
    distvect = np.array([dx, dy, dz])
    distnorm = np.linalg.norm(distvect)
    distances.append(distnorm)
    # TODO: update these with how HOOMD does it backend: https://github.com/glotzerlab/hoomd-blue/blob/79431aa329f1cc5ff40707a059cfa9909a902675/hoomd/md/EvaluatorPairGB.h#L214
    # (grab bottom row of rotation matrix -> a3, b3) Get rotation matrices from rowan.to_matrix(orientation)
    ei = frame.particles.orientation[0][1:]
    ei = ei / np.linalg.norm(ei)
    ej = frame.particles.orientation[1][1:]
    ej = ej / np.linalg.norm(ej)
    gbdists.append(distconvert(distvect, lperp, lpar, ei, ej))
    phidists.append(phiconvert(distvect, lperp, lpar, ei, ej))
    energies.append(frame.log['md/pair/aniso/GayBerne/energy'])
plt.figure()
plt.plot(distances, label = 'center-to-center distances')
plt.plot(gbdists, label = 'adjusted distances ($\zeta$)')
plt.plot(phidists, label = 'scaled potential distances ($\Phi$, equation 14)')
plt.plot(energies, label='Gay-Berne Energies')
plt.vlines([297, 826], -10,10)
plt.xlabel('timestep')
plt.ylabel('distance (sigma)')
plt.plot(2.5 * np.ones_like(distances), ls = '--', label = 'cutoff radius')
plt.plot(((2.5 - sigmax + sigmin)/sigmin) * np.ones_like(distances), ls = '--', label = 'adjusted cutoff radius ($\zeta_{cut}$')
plt.plot(2.5 / sigmax * np.ones_like(distances), ls = '--', label = '$\Phi$ cutoff radius')
plt.ylim(-5, 5)
plt.legend()
plt.savefig('distance_rcut_plot.png', dpi=300)
