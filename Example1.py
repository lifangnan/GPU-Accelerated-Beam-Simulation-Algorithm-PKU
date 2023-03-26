import pyBeamSim

ref_energy = 5
num_particle = 10240
simulator = pyBeamSim.BeamSimulator()
# simulator.free_beam()
simulator.init_beam(num_particle, 938.272046, 1.0, 0.0)
simulator.set_beamTwiss(0, 0.003, 0.0001, 0, 0.003, 0.0001, 0, 8, 3.1415926e-11, 0, ref_energy, 500, 1)
simulator.save_initial_beam()

simulator.load_Beamline_From_DatFile("./model/clapa1_5MeV.dat", 5)

envelope = simulator.simulate_and_getEnvelope_CPU()

simulator.plot_envelope(envelope)
simulator.plot_beam_phase_map()

simulator.UpdateBeamParameters_CPU()
print("Avg X (m):", simulator.getAvgX())
print("Sig X (rad):", simulator.getSigX())
print("Emittance X (m*rad):", simulator.getEmitX())
print("Unlost Number of Particles:", simulator.getGoodNum())