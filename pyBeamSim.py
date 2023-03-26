from cgitb import enable
from ctypes import * 
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import json

dll = CDLL("./HpsimLib.dll")

# 定义参数和返回值的类型
dll.init_beam.argtypes = [c_int, c_double, c_double, c_double]

dll.init_beam_from_file.argtypes = [c_char_p]
# dll.update_beam_from_file.argtypes = [c_char_p]

dll.beam_print_to_file.argtypes =[c_char_p, c_char_p]

dll.set_beamTwiss.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_uint]

dll.move_beam_center.argtypes = [c_double, c_double]

dll.set_beam_energy_spectrum.argtypes = [c_double, c_double, c_double, c_double, c_double]

dll.GetParticlesState.argtypes = [c_bool]
dll.GetParticlesState.restype = c_char_p

dll.getAvgX.restype = c_double
dll.getAvgY.restype = c_double
dll.getAvgXp.restype = c_double
dll.getAvgYp.restype = c_double
dll.getSigX.restype = c_double
dll.getSigY.restype = c_double
dll.getSigXp.restype = c_double
dll.getSigYp.restype = c_double
dll.getEmitX.restype = c_double
dll.getEmitY.restype = c_double
dll.getEmitZ.restype = c_double
dll.getGoodNum.restype = c_int

# dll.getBeamAvgx.restype = c_double
# dll.getBeamAvgy.restype = c_double
# dll.getBeamSigx.restype = c_double
# dll.getBeamSigy.restype = c_double
dll.getBeamMaxx.restype = c_double
dll.getBeamMaxy.restype = c_double


dll.add_Drift.argtypes = [c_char_p, c_double, c_double]
dll.add_Bend.argtypes = [c_char_p, c_double, c_double, c_double, c_double, c_double, c_double, c_double, c_double]
dll.add_Quad.argtypes = [c_char_p, c_double, c_double, c_double]
dll.add_Solenoid.argtypes = [c_char_p, c_double, c_double, c_double]
dll.add_ApertureRectangular.argtypes = [c_char_p, c_double, c_double, c_double, c_double]
dll.add_ApertureCircular.argtypes = [c_char_p, c_double]

# dll.init_database.argtypes = [c_char_p]

dll.load_Beamline_From_DatFile.argtypes = [c_char_p, c_double]

dll.get_Beamline_ElementNames.restype = c_char_p
dll.get_Beamline_ElementTypes.restype = c_char_p
dll.get_Beamline_ElementLengths.restype = c_char_p
dll.get_Beamline_ElementApertures.restype = c_char_p

dll.init_spacecharge.argtypes = [c_uint, c_uint, c_int]

dll.init_simulator.argtypes = [c_bool]
dll.simulate_from_to.argtypes = [c_char_p, c_char_p]

dll.simulate_and_getEnvelope.argtypes = [c_bool]
dll.simulate_and_getEnvelope.restype = c_char_p
dll.simulate_and_getEnvelope_CPU.argtypes = [c_bool]
dll.simulate_and_getEnvelope_CPU.restype = c_char_p
dll.simulate_all.argtypes = [c_bool]

dll.set_magnet_with_index.argtypes = [c_int, c_double]
dll.set_magnet_with_name.argtypes = [c_char_p, c_double]
dll.move_magnet_with_index.argtypes = [c_int, c_double]
dll.move_magnet_with_name.argtypes = [c_char_p, c_double]

dll.UpdateBeamParameters_CPU.restype = c_float

class BeamSimulator():
    def __init__(self):
        pass

    def init_Default_Simulator(self):
        dll.default_init()
    
    def init_beam(self, particle_num, rest_energy, charge, current):
        dll.init_beam(particle_num, rest_energy, charge, current)
    
    def init_beam_from_file(self, filename):
        dll.init_beam_from_file(filename.encode())

    # def update_beam_from_file(self, filename):
    #     dll.update_beam_from_file(filename.encode())
    
    def beam_print_to_file(self, filePath, comment = "CLAPA"):
        dll.beam_print_to_file(filePath.encode(), comment.encode())
    
    def set_beamTwiss(self, r_ax, r_bx, r_ex, r_ay, r_by, r_ey, r_az, r_bz, r_ez, r_sync_phi, r_sync_w, r_freq, r_seed = 1):
        dll.set_beamTwiss(r_ax, r_bx, r_ex, r_ay, r_by, r_ey, r_az, r_bz, r_ez, r_sync_phi, r_sync_w, r_freq, r_seed)
    
    def move_beam_center(self, dx, dy):
        dll.move_beam_center(dx, dy)

    def set_beam_energy_spectrum(self, A=25, B=0.4, C=0, min_energy=3, max_energy=10):
        dll.set_beam_energy_spectrum(A, B, C, min_energy, max_energy)

    def save_initial_beam(self):
        dll.save_initial_beam()
    
    def restore_initial_beam(self):
        dll.restore_initial_beam()

    def GetParticlesState(self, only_good_particles = True):
        json_particleState = dll.GetParticlesState(only_good_particles).decode()
        return json.loads(json_particleState)


    # def getBeamAvgx(self):
    #     return dll.getBeamAvgx()
    
    # def getBeamAvgy(self):
    #     return dll.getBeamAvgy()

    # def getBeamSigx(self):
    #     return dll.getBeamSigx()

    # def getBeamSigy(self):
    #     return dll.getBeamSigy()

    def getBeamMaxx(self):
        return dll.getBeamMaxx()

    def getBeamMaxy(self):
        return dll.getBeamMaxy()

    def UpdateBeamParameters(self):
        dll.UpdateBeamParameters()

    def UpdateBeamParameters_CPU(self):
        return dll.UpdateBeamParameters_CPU()
    
    def getAvgX(self):
        return dll.getAvgX()

    def getAvgY(self):
        return dll.getAvgY()
    
    def getAvgXp(self):
        return dll.getAvgXp()

    def getAvgYp(self):
        return dll.getAvgYp()

    def getAvgEnergy(self):
        return dll.getAvgEnergy()
    
    def getSigX(self):
        return dll.getSigX()
    
    def getSigY(self):
        return dll.getSigY()
    
    def getSigXp(self):
        return dll.getSigXp()
    
    def getSigYp(self):
        return dll.getAvgYp()
    
    def getSigEnergy(self):
        return dll.getSigEnergy()
    
    def getEnergySpread(self):
        return dll.getEnergySpread()
    
    def getEmitX(self):
        return dll.getEmitX()
    
    def getEmitY(self):
        return dll.getEmitY()
    
    def getEmitZ(self):
        return dll.getEmitZ()
    
    def getGoodNum(self):
        return dll.getGoodNum()
    

    def free_beam(self):
        dll.free_beam()


    def init_Beamline(self):
        dll.init_Beamline()
    
    def add_Drift(self, ID, Length, Aperture):
        dll.add_Drift(ID.encode(), Length, Aperture)
    
    def add_Bend(self, ID, Length, Aperture, Angle, AngleIn, AngleOut, DefaultField, Charge, RestEnergy):
        dll.add_Bend(ID.encode(), Length, Aperture, Angle, AngleIn, AngleOut, DefaultField, Charge, RestEnergy)
    
    def add_Quad(self, ID, Length, Aperture, FieldGradient):
        dll.add_Quad(ID.encode(), Length, Aperture, FieldGradient)
    
    def add_Solenoid(self, ID, Length, Aperture, FieldGradient):
        dll.add_Solenoid(ID.encode(), Length, Aperture, FieldGradient)
    
    def add_ApertureRectangular(self, ID, XLeft, XRight, YBottom, YTop):
        dll.add_ApertureRectangular(ID.encode(), XLeft, XRight, YBottom, YTop)
    
    def add_ApertureCircular(self, ID, Aperture):
        dll.add_ApertureCircular(ID.encode(), Aperture)


    # def load_Beamline_From_Sqlite(self, DB_Path):
    #     dll.init_database(DB_Path.encode())
    #     dll.init_beamline_from_DB()

    def load_Beamline_From_DatFile(self, filename, ReferenceEnergy = 100):
        dll.load_Beamline_From_DatFile(filename.encode(), ReferenceEnergy)

    def get_Beamline_ElementNames(self):
        names_str = (dll.get_Beamline_ElementNames()).decode()
        names_list = names_str.split(",")
        return names_list
    
    def get_Beamline_ElementTypes(self):
        types_str = (dll.get_Beamline_ElementTypes()).decode()
        types_list = types_str.split(",")
        return types_list
    
    def get_Beamline_ElementLengths(self):
        lengths_str = (dll.get_Beamline_ElementLengths()).decode()
        lengths_list = lengths_str.split(",")
        lengths_list = [float(item) for item in lengths_list]
        return np.array(lengths_list)

    def get_Beamline_ElementApertures(self):
        apertures_str = (dll.get_Beamline_ElementApertures()).decode()
        apertures_list = apertures_str.split(",")
        apertures_list = [float(item) for item in apertures_list]
        return np.array(apertures_list)


    def init_spacecharge(self, r_nr = 32, r_nz = 128, r_adj_bunch = 3):
        dll.init_spacecharge(r_nr, r_nz, r_adj_bunch)
    

    def init_simulator(self, use_spacecharge = False):
        dll.init_simulator(use_spacecharge)
    
    def simulate_from_to(self, begin_element_ID, end_element_ID):
        dll.simulate_from_to(begin_element_ID.encode(), end_element_ID.encode())

    def simulate_and_getEnvelope(self, use_spacecharge = False):
        envelope = dll.simulate_and_getEnvelope(use_spacecharge)
        
        envelope_json = json.loads(envelope)

        return envelope_json

    def simulate_and_getEnvelope_CPU(self, use_spacecharge = False):
        envelope = dll.simulate_and_getEnvelope_CPU(use_spacecharge)
        
        envelope_json = json.loads(envelope)

        return envelope_json
    
    def simulate_all(self, use_spacecharge = False):
        dll.simulate_all(use_spacecharge)

    def set_magnet_with_index(self, magnet_index, field_or_angle):
        dll.set_magnet_with_index(magnet_index, field_or_angle)
    
    def set_magnet_with_name(self, element_name, field_or_angle):
        dll.set_magnet_with_name(element_name.encode(), field_or_angle)

    def move_magnet_with_index(self, magnet_index, move_delta_z):
        dll.move_magnet_with_index(magnet_index, move_delta_z)
    
    def move_magnet_with_name(self, element_name, move_delta_z):
        dll.move_magnet_with_name(element_name.encode(), move_delta_z)
        

    # def plot_envelope(self, envelope):
    #     element_types = self.get_Beamline_ElementTypes()
    #     element_lengths = self.get_Beamline_ElementLengths()
    #     position_start = np.array([])
    #     position_end = np.array([])
    #     for i in range(element_lengths.shape[0]):
    #         position_start = np.append(position_start, element_lengths[:i].sum())
    #         position_end = np.append(position_end, position_start[i] + element_lengths[i])

    #     plt.figure(figsize=(13,3))
    #     plt.plot(envelope[:,0], envelope[:,1])
    #     plt.plot(envelope[:,0], envelope[:,2])

    #     max_enve = envelope[:, [1,2]].max().max()

    #     deviceType_list = ["Dipole", "Solenoid", "Quad"]
    #     for i in range(len(element_types)):
    #         if element_types[i] in deviceType_list:
    #             element_start = position_start[i]
    #             element_end = position_end[i]
    #             if element_types[i] == "Dipole":
    #                 plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'pink', alpha = 0.9)
    #             elif element_types[i] == "Solenoid":
    #                 plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'green', alpha = 0.3)
    #             elif element_types[i] == "Quad":
    #                 plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'blue', alpha = 0.3)   

    #     plt.show()

    def plot_envelope(self, envelope_dict):
        element_types = self.get_Beamline_ElementTypes()
        element_lengths = self.get_Beamline_ElementLengths()
        position_start = np.array([])
        position_end = np.array([])
        for i in range(element_lengths.shape[0]):
            position_start = np.append(position_start, element_lengths[:i].sum())
            position_end = np.append(position_end, position_start[i] + element_lengths[i])

        
        plt.figure(figsize=(13,3))
        plt.plot(envelope_dict['z'], envelope_dict['Sig_x'])
        plt.plot(envelope_dict['z'], envelope_dict['Sig_y'])
        plt.xlabel("z/m")
        plt.ylabel("RMS/m")
        plt.legend(['RMS of x', 'RMS of y'])

        max_enve = max(max(envelope_dict['Sig_x']), max(envelope_dict['Sig_y']))
        
        deviceType_list = ["Dipole", "Solenoid", "Quad"]
        for i in range(len(element_types)):
            if element_types[i] in deviceType_list:
                element_start = position_start[i]
                element_end = position_end[i]
                if element_types[i] == "Dipole":
                    plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'pink', alpha = 0.9)
                elif element_types[i] == "Solenoid":
                    plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'green', alpha = 0.3)
                elif element_types[i] == "Quad":
                    plt.fill_between([element_start, element_end],0, max_enve,facecolor = 'blue', alpha = 0.3)   

        plt.show()

        plt.figure(figsize=(13,3))
        plt.plot(envelope_dict['z'], envelope_dict['Avg_x'])
        plt.plot(envelope_dict['z'], envelope_dict['Avg_y'])

        max_avg = max(max(envelope_dict['Avg_x']), max(envelope_dict['Avg_y']))
        min_avg = min(min(envelope_dict['Avg_x']), min(envelope_dict['Avg_y']))
        for i in range(len(element_types)):
            if element_types[i] in deviceType_list:
                element_start = position_start[i]
                element_end = position_end[i]
                if element_types[i] == "Dipole":
                    plt.fill_between([element_start, element_end], min_avg, max_avg,facecolor = 'pink', alpha = 0.9)
                elif element_types[i] == "Solenoid":
                    plt.fill_between([element_start, element_end], min_avg, max_avg,facecolor = 'green', alpha = 0.3)
                elif element_types[i] == "Quad":
                    plt.fill_between([element_start, element_end], min_avg, max_avg,facecolor = 'blue', alpha = 0.3)   
        plt.xlabel("z/m")
        plt.ylabel("Average Position/m")
        plt.legend(['Average Position of x', 'Average Position of y'])
        plt.show()

        plt.figure(figsize=(13,3))
        plt.plot(envelope_dict['z'], envelope_dict['Good Particle Number'])

        max_number = max(envelope_dict['Good Particle Number'])
        min_number = min(envelope_dict['Good Particle Number'])
        for i in range(len(element_types)):
            if element_types[i] in deviceType_list:
                element_start = position_start[i]
                element_end = position_end[i]
                if element_types[i] == "Dipole":
                    plt.fill_between([element_start, element_end], min_number, max_number,facecolor = 'pink', alpha = 0.9)
                elif element_types[i] == "Solenoid":
                    plt.fill_between([element_start, element_end], min_number, max_number,facecolor = 'green', alpha = 0.3)
                elif element_types[i] == "Quad":
                    plt.fill_between([element_start, element_end], min_number, max_number,facecolor = 'blue', alpha = 0.3)   
        plt.xlabel("z/m")
        plt.ylabel("Good Particles Number")
        plt.show()


    # def plot_beam(self):
    #     cwd = os.getcwd()
    #     self.beam_print_to_file(cwd + "\\temp_Beam")
    #     beam_data = np.loadtxt(cwd + "\\temp_Beam")

    #     beam_data = beam_data[beam_data[:,-2]==0, :]
    #     # intibeam_df = pd.DataFrame(columns=['x','y'])
    #     # intibeam_df['x'] = beam_data[:, 0]
    #     # intibeam_df['y'] = beam_data[:, 2]
    #     # sns.jointplot(x="x", y="y", data=intibeam_df, kind="kde", levels=50, fill=True, cmap='binary')

    #     plt.figure(figsize=[15,5])
    #     plt.subplot(1,3,1)
    #     plt.scatter(beam_data[:, 0], beam_data[:, 2], s=1)
    #     plt.axis("equal")
    #     plt.xlabel("x")
    #     plt.ylabel("y")
    #     plt.title("x-y")

    #     plt.subplot(1,3,2)
    #     plt.scatter(beam_data[:, 0], beam_data[:, 1], s=1)
    #     plt.xlabel("x")
    #     plt.ylabel("px")
    #     plt.title("x-px")

    #     plt.subplot(1,3,3)
    #     plt.scatter(beam_data[:, 2], beam_data[:, 3], s=1)
    #     plt.xlabel("y")
    #     plt.ylabel("py")
    #     plt.title("y-py")

    #     plt.show()

    #     os.remove(cwd + "\\temp_Beam")

    def plot_beam_phase_map(self):
        self.UpdateBeamParameters()
        particles = self.GetParticlesState()
        plt.figure(figsize=[15,5])
        plt.subplot(1,3,1)
        plt.scatter(particles['x'], particles['y'], s=1)
        plt.axis("equal")
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("x-y")

        plt.subplot(1,3,2)
        plt.scatter(particles['x'], particles['xp'], s=1)
        plt.xlabel("x(m)")
        plt.ylabel("xp(rad)")
        plt.title("x-xp")

        plt.subplot(1,3,3)
        plt.scatter(particles['y'], particles['yp'], s=1)
        plt.xlabel("y(m)")
        plt.ylabel("yp(rad)")
        plt.title("y-yp")
        plt.show()

    
    def dllTest(self):
        dll.test()