#define EXPORT __declspec(dllexport)

#include "beam.h"
#include "beam_cu.h"
#include "beamline.h"
#include "beamline_element.h"
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<map>
#include<cuda.h>
#include<string>
#include <cuda_runtime_api.h>
#include<algorithm>

#include "init.h"
#include "sql_utility.h"
#include "constant.h"

#include "space_charge.h"
#include "simulation_engine.h"
#include "simulation_engine_cu.h"
#include "plot_data.h"

#include "json.hpp"
#include "timer.h"

using namespace std;
using json = nlohmann::json;

// some global variables
Beam* beam;
Beam* beam1;
BeamLine beamline;
Scheff* spacecharge;
SimulationEngine* simulator;
int beamline_size;
int envelope_size;
int particle_size;

// double current_position = 0;
double avg_x = 0, avg_y = 0, avg_xp = 0, avg_yp = 0;
double sig_x = 0, sig_y = 0, sig_xp = 0, sig_yp = 0;
double emit_x = 0, emit_y = 0, emit_z = 0, emit_all = 0;
double emit_x_trace = 0, emit_y_trace = 0;
double alpha_x = 0, alpha_y = 0, alpha_z = 0;
double beta_x = 0, beta_y = 0, beta_z = 0;
double gamma_x = 0, gamma_y = 0, gamma_z = 0;
double avg_w = 0, sig_energy = 0, energy_spread = 0;
int count_goodnum = 0;

char *rt_names, *rt_types, *rt_lengths, *rt_char_data;
double **rt_envelope;

vector<double> vec_energy;
vector<double> energy_distribution;

double get_a_random_energy(){
    double rand_num = rand()/double(RAND_MAX) * energy_distribution.back();
    for(int i=0; i<vec_energy.size(); i++){
        if(energy_distribution[i] >= rand_num){
            return vec_energy[i];
        }
    }
    return NULL;
}

extern "C" {
    EXPORT void init_beam(int particle_num, double rest_energy, double charge, double current){
        SetGPU(0);
        beam = new Beam(particle_num, rest_energy, charge, current);
    }

    EXPORT void init_beam_from_file(char* file_path){
        if(beam){
            beam->InitBeamFromFile(string(file_path));
            // beam_init->InitBeamFromFile(file_path);
        }else{
            cout<<"Please use init_beam() first."<<endl;
        }
    }  

    // EXPORT void update_beam_from_file(char* file_path){
    //     if(beam){
    //         beam->UpdateBeamFromFile(string(file_path));
    //         // beam_init->InitBeamFromFile(file_path);
    //     }else{
    //         cout<<"Please use init_beam() first."<<endl;
    //     }
    // }  

    EXPORT void beam_print_to_file(char* file_path, char* comment = "CLAPA"){
        if(beam){
            beam->PrintToFile(file_path, comment);
        }
    }  

    EXPORT void set_beamTwiss(double r_ax, double r_bx, double r_ex, 
                double r_ay, double r_by, double r_ey, double r_az, double r_bz, double r_ez,
                double r_sync_phi, double r_sync_w, double r_freq, unsigned int r_seed = 1){
        // if(beam){
        beam->InitWaterbagBeam(r_ax, r_bx, r_ex, r_ay, r_by, r_ey, r_az, r_bz, r_ez, r_sync_phi, r_sync_w, r_freq, r_seed);
        // }
    }  

    // 移动靶点位置，即移动粒子束中心位置，单位毫米mm
    EXPORT void move_beam_center(double dx, double dy){
        uint num_particle = beam->num_particle;
        double* x_h = new double[num_particle];
        double* xp_h = new double[num_particle];
        double* y_h = new double[num_particle];
        double* yp_h = new double[num_particle];
        double* phi_h = new double[num_particle];
        double* w_h = new double[num_particle];
        uint* loss_h = new uint[num_particle];
        uint* lloss_h = new uint[num_particle];
        uint* num_loss_h = new uint;

        CopyBeamFromDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);

        for(int i = 0; i < num_particle; i++){
            x_h[i] += dx/1000;
            y_h[i] += dy/1000;
        }

        UpdateBeamOnDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h);

        delete [] x_h; delete [] xp_h;
        delete [] y_h; delete [] yp_h;
        delete [] phi_h; delete [] w_h; 
        delete [] loss_h; delete [] lloss_h;
        delete num_loss_h;
    }


    EXPORT void set_beam_energy_spectrum(double A, double B, double C, double min_energy, double max_energy){
        vec_energy.clear();
        energy_distribution.clear();
        double dE = 0.001, temp_E = min_energy;
        while (temp_E <= max_energy){
            vec_energy.push_back(temp_E);

            if(temp_E == min_energy){
                energy_distribution.push_back(0);
            }else{
                energy_distribution.push_back( energy_distribution.back() + exp( A - B * temp_E) + C );
            }

            temp_E += dE;
        }

        srand(time(0));

        uint num_particle = beam->num_particle;
        double* x_h = new double[num_particle];
        double* xp_h = new double[num_particle];
        double* y_h = new double[num_particle];
        double* yp_h = new double[num_particle];
        double* phi_h = new double[num_particle];
        double* w_h = new double[num_particle];
        uint* loss_h = new uint[num_particle];
        uint* lloss_h = new uint[num_particle];
        uint* num_loss_h = new uint;

        CopyBeamFromDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);

        for(int i = 0; i < num_particle; i++){
            w_h[i] = get_a_random_energy();
        }

        UpdateBeamOnDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h);

        delete [] x_h; delete [] xp_h;
        delete [] y_h; delete [] yp_h;
        delete [] phi_h; delete [] w_h; 
        delete [] loss_h; delete [] lloss_h;
        delete num_loss_h;
    }


    EXPORT void save_initial_beam(){
        if(beam){
            beam->SaveInitialBeam();
        }
    }

    EXPORT void restore_initial_beam(){
        if(beam){
            beam->RestoreInitialBeam();
        }
    }


    EXPORT char* GetParticlesState(bool only_good_particles){
        vector<uint> loss = beam->GetLoss();
        uint num_particle = beam->num_particle;
        double* x_h = new double[num_particle];
        double* xp_h = new double[num_particle];
        double* y_h = new double[num_particle];
        double* yp_h = new double[num_particle];
        double* phi_h = new double[num_particle];
        double* w_h = new double[num_particle];
        uint* loss_h = new uint[num_particle];
        uint* lloss_h = new uint[num_particle];
        uint* num_loss_h = new uint;

        CopyBeamFromDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);

        vector<double> vec_x, vec_xp, vec_y, vec_yp, vec_phi, vec_energy;
        vector<uint> vec_loss;

        for(int i = 0; i < num_particle; i++){
            if(only_good_particles && loss[i] == 0){
                vec_x.push_back(x_h[i]);
                vec_xp.push_back(xp_h[i]);
                vec_y.push_back(y_h[i]);
                vec_yp.push_back(yp_h[i]);
                vec_phi.push_back(phi_h[i]);
                vec_energy.push_back(w_h[i]);
                vec_loss.push_back(loss[i]);
            }else if(!only_good_particles){
                vec_x.push_back(x_h[i]);
                vec_xp.push_back(xp_h[i]);
                vec_y.push_back(y_h[i]);
                vec_yp.push_back(yp_h[i]);
                vec_phi.push_back(phi_h[i]);
                vec_energy.push_back(w_h[i]);
                vec_loss.push_back(loss[i]);
            }
        }

        json rt_data;
        json x_json(vec_x); rt_data["x"] = x_json;
        json xp_json(vec_xp); rt_data["xp"] = xp_json;
        json y_json(vec_y); rt_data["y"] = y_json;
        json yp_json(vec_yp); rt_data["yp"] = yp_json;
        json phi_json(vec_phi); rt_data["phi"] = phi_json;
        json energy_json(vec_energy); rt_data["energy"] = energy_json;
        json loss_json(vec_loss); rt_data["loss"] = loss_json;

        string data_string = rt_data.dump();

        int charlength = strlen(data_string.c_str()) + 1;
        delete rt_char_data;
        rt_char_data = new char[charlength];
        strcpy_s(rt_char_data, charlength, data_string.c_str());

        return rt_char_data;
    }

    // EXPORT double getBeamAvgx(){
    //     vector<uint> loss = beam->GetLoss();
    //     vector<double> x = beam->GetX();
    //     double sumX = 0;
    //     int count_goodnum = 0;

    //     for(int i=0; i<x.size(); i++){
    //         if(loss[i] == 0){
    //             count_goodnum++;
    //             sumX += x[i];
    //         }
    //     }
    //     return sumX / count_goodnum;
    // }

    // EXPORT double getBeamAvgy(){
    //     vector<uint> loss = beam->GetLoss();
    //     vector<double> y = beam->GetY();
    //     double sumY = 0;
    //     int count_goodnum = 0;

    //     for(int i=0; i<y.size(); i++){
    //         if(loss[i] == 0){
    //             count_goodnum++;
    //             sumY += y[i];
    //         }
    //     }
    //     return sumY / count_goodnum;
    // }
    
    // EXPORT double getBeamSigx(){
    //     vector<uint> loss = beam->GetLoss();
    //     vector<double> x = beam->GetX();
    //     double sumX = 0;
    //     int count_goodnum = 0;

    //     for(int i=0; i<x.size(); i++){
    //         if(loss[i] == 0){
    //             count_goodnum++;
    //             sumX += x[i];
    //         }
    //     }
    //     double avgX = sumX / count_goodnum;
    //     double temp_sum = 0;
    //     for(int i=0; i<x.size(); i++){
    //         if(loss[i] == 0){
    //             temp_sum += (x[i] - avgX)*(x[i] - avgX);
    //         }
    //     }
    //     if(count_goodnum > 1){
    //         double stdX = sqrt(temp_sum/(count_goodnum-1));
    //         return stdX;
    //     }else{
    //         return 0.0;
    //     } 
    // }

    // EXPORT double getBeamSigy(){
    //     vector<uint> loss = beam->GetLoss();
    //     vector<double> Y = beam->GetY();
    //     double sumY = 0;
    //     int count_goodnum = 0;

    //     for(int i=0; i<Y.size(); i++){
    //         if(loss[i] == 0){
    //             count_goodnum++;
    //             sumY += Y[i];
    //         }
    //     }
    //     double avgY = sumY / count_goodnum;
    //     double temp_sum = 0;
    //     for(int i=0; i<Y.size(); i++){
    //         if(loss[i] == 0){
    //             temp_sum += (Y[i] - avgY)*(Y[i] - avgY);
    //         }
    //     }
    //     if(count_goodnum > 1){
    //         double stdY = sqrt(temp_sum/(count_goodnum-1));
    //         return stdY;
    //     }else{
    //         return 0.0;
    //     } 
    // }

    EXPORT double getBeamMaxx(){
        vector<uint> loss = beam->GetLoss();
        vector<double> x = beam->GetX();
        double maxX = 0, tempX = 0;

        for(int i=0; i<x.size(); i++){
            if(loss[i] == 0){
                tempX = abs(x[i]);
                if(tempX > maxX) maxX = tempX;
            }
        }
        return maxX;
    }

    EXPORT double getBeamMaxy(){
        vector<uint> loss = beam->GetLoss();
        vector<double> y = beam->GetY();
        double maxY = 0, tempY = 0;

        for(int i=0; i<y.size(); i++){
            if(loss[i] == 0){
                tempY = abs(y[i]);
                if(tempY > maxY) maxY = tempY;
            }
        }
        return maxY;
    }


    EXPORT void UpdateBeamParameters(){
        cudaEvent_t start, stop;  
        StartTimer(&start, &stop);
        beam->UpdateAvgX();
        // StopTimer(&start, &stop, "UpdateAvgX: ");
        beam->UpdateAvgY();
        beam->UpdateAvgXp();
        beam->UpdateAvgYp();
        beam->UpdateAvgW();
        // StartTimer(&start, &stop);
        beam->UpdateSigX();
        // StopTimer(&start, &stop, "UpdateSigX: ");
        beam->UpdateSigY();
        beam->UpdateSigXp();
        beam->UpdateSigYp();
        beam->UpdateSigW();
        // StartTimer(&start, &stop);
        beam->UpdateEmittance();
        // StopTimer(&start, &stop, "UpdateEmittance: ");
        // StartTimer(&start, &stop);
        beam->UpdateLoss();
        // StopTimer(&start, &stop, "UpdateLoss: ");
        // StartTimer(&start, &stop);
        beam->UpdateGoodParticleCount();
        // StopTimer(&start, &stop, "UpdateGoodParticleCount: ");

        // StartTimer(&start, &stop);
        avg_x = beam->GetAvgX();
        // StopTimer(&start, &stop, "GetAvgX: ");
        avg_y = beam->GetAvgY();
        avg_xp = beam->GetAvgXp();
        avg_yp = beam->GetAvgYp();
        // StartTimer(&start, &stop);
        emit_x = beam->GetEmittanceX();
        // StopTimer(&start, &stop, "GetEmittanceX: ");
        // StartTimer(&start, &stop);
        emit_y = beam->GetEmittanceY();
        // StopTimer(&start, &stop, "GetEmittanceY: ");
        // StartTimer(&start, &stop);
        emit_z = beam->GetEmittanceZ();
        // StopTimer(&start, &stop, "GetEmittanceZ: ");
        // StartTimer(&start, &stop);
        sig_x = beam->GetSigX();
        // StopTimer(&start, &stop, "GetSigX: ");
        sig_y = beam->GetSigY();
        sig_xp = beam->GetSigXp();
        sig_yp = beam->GetSigYp();
        sig_energy = beam->GetSigW();

        
        // StartTimer(&start, &stop);
        count_goodnum = beam->GetGoodParticleNum();
        // StopTimer(&start, &stop, "GetGoodParticleNum: ");

        energy_spread = sig_energy/avg_w * 2.35482005;

        StopTimer(&start, &stop, "UpdateBeamParameters: ");
    }

    // EXPORT float UpdateBeamParameters_CPU(){
    //     cudaEvent_t start, stop;  
    //     // StartTimer(&start, &stop);

    //     vector<uint> loss = beam->GetLoss();
    //     uint num_particle = beam->num_particle;
    //     double* x_h = new double[num_particle];
    //     double* xp_h = new double[num_particle];
    //     double* y_h = new double[num_particle];
    //     double* yp_h = new double[num_particle];
    //     double* phi_h = new double[num_particle];
    //     double* w_h = new double[num_particle];
    //     uint* loss_h = new uint[num_particle];
    //     uint* lloss_h = new uint[num_particle];
    //     uint* num_loss_h = new uint;

    //     double* px_h = new double[num_particle];
    //     double* py_h = new double[num_particle];

    //     // StartTimer(&start, &stop);
    //     cudaEventCreate(&start);
    //     cudaEventCreate(&stop);
    //     cudaEventRecord(start, 0);
    //     CopyBeamFromDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);
    //     // StopTimer(&start, &stop, "copy beam from GPU: ");
    //     cudaThreadSynchronize();
    //     cudaEventRecord(stop, 0);   
    //     cudaEventSynchronize(stop);
    //     float elapsedTime_g;
    //     cudaEventElapsedTime(&elapsedTime_g, start, stop); // 計算時間差
    //     printf("Global memory processing time: %f (ms)\n", elapsedTime_g);

    //     // calculate mean
    //     double sum_x = 0, sum_y = 0 , sum_xp = 0, sum_yp = 0;
    //     double sum_xx = 0, sum_xpxp = 0, sum_xxp = 0;
    //     double sum_yy = 0, sum_ypyp = 0, sum_yyp = 0;
    //     double sum_zz = 0, sum_zpzp = 0, sum_zzp = 0;
    //     double sum_w2 = 0, sum_phi2 = 0, sum_phi_w = 0, sum_phi = 0;
    //     double sum_w = 0;
    //     count_goodnum = 0;
    //     for(int i = 0; i < num_particle; i++){
    //         if(loss[i] == 0){
    //             sum_x += x_h[i];
    //             sum_y += y_h[i];
    //             sum_xp += xp_h[i];
    //             sum_yp += yp_h[i];
    //             sum_w += w_h[i];
    //             sum_phi += phi_h[i];

    //             sum_xx += x_h[i] * x_h[i];
    //             sum_xpxp += xp_h[i] * xp_h[i];
    //             sum_xxp += x_h[i] * xp_h[i];

    //             sum_yy += y_h[i] * y_h[i];
    //             sum_ypyp += yp_h[i] * yp_h[i];
    //             sum_yyp += y_h[i] * yp_h[i];

    //             sum_w2 += w_h[i] * w_h[i];
    //             sum_phi2 += phi_h[i] * phi_h[i];
    //             sum_phi_w += phi_h[i] * w_h[i];

    //             // sum_zz += z_h[i] * z_h[i];
    //             // sum_zpzp += zp_h[i] * zp_h[i];
    //             // sum_zzp += z_h[i] * zp_h[i];

    //             count_goodnum++;
    //         }
    //     }

    //     if(count_goodnum <= 2){
    //         avg_x = NULL;
    //         avg_y = NULL;
    //         avg_xp = NULL;
    //         avg_yp = NULL;
    //         emit_x = NULL;
    //         emit_y = NULL;
    //         sig_x = NULL;
    //         sig_y = NULL;
    //         sig_xp = NULL;
    //         sig_yp = NULL;
    //         sig_energy = NULL;
    //         return 0;
    //     }

    //     avg_x = sum_x / count_goodnum;
    //     avg_y = sum_y / count_goodnum;
    //     avg_xp = sum_xp / count_goodnum;
    //     avg_yp = sum_yp / count_goodnum;

    //     double avg_x2 = sum_xx / count_goodnum;
    //     double avg_xp2 = sum_xpxp / count_goodnum;
    //     double avg_xxp = sum_xxp / count_goodnum;
    //     double avg_y2 = sum_yy / count_goodnum;
    //     double avg_yp2 = sum_ypyp / count_goodnum;
    //     double avg_yyp = sum_yyp / count_goodnum;
    //     double avg_phi = sum_phi / count_goodnum;
    //     double avg_w2 = sum_w2 / count_goodnum;
    //     double avg_phi2 = sum_phi2 / count_goodnum;
    //     double avg_phi_w = sum_phi_w / count_goodnum;

    //     avg_w = sum_w / count_goodnum;

    //     double gamma1 = avg_w / beam->mass;
    //     double betagamma = sqrt(gamma1 * (gamma1 + 2.0));

    //     emit_x = betagamma * sqrt((avg_x2 - avg_x * avg_x) * (avg_xp2 - avg_xp * avg_xp) - 
    //         (avg_xxp - avg_x * avg_xp) * (avg_xxp - avg_x * avg_xp));
    //     emit_y = betagamma * sqrt((avg_y2 - avg_y * avg_y) * (avg_yp2 - avg_yp * avg_yp) - 
    //         (avg_yyp - avg_y * avg_yp) * (avg_yyp - avg_y * avg_yp));

    //     double phi_var = avg_phi2 - avg_phi * avg_phi;
    //     double w_var = avg_w2 - avg_w * avg_w;
    //     double phi_w_covar = avg_phi_w - avg_phi * avg_w;
    //     emit_z = sqrt(phi_var * w_var - phi_w_covar * phi_w_covar);

    //     // calculate sigma
    //     sum_x = 0; sum_y = 0; sum_xp = 0; sum_yp = 0, sum_w = 0;
    //     for(int i = 0; i < num_particle; i++){
    //         if(loss[i] == 0){
    //             sum_x += (x_h[i] - avg_x)*(x_h[i] - avg_x);
    //             sum_y += (y_h[i] - avg_y)*(y_h[i] - avg_y);
    //             sum_xp += (xp_h[i] - avg_xp)*(xp_h[i] - avg_xp);
    //             sum_yp += (yp_h[i] - avg_yp)*(yp_h[i] - avg_yp);
    //             sum_w += (w_h[i] - avg_w)*(w_h[i] - avg_w);
    //         }
    //     }
    //     if(count_goodnum > 1){
    //         sig_x = sqrt(sum_x/(count_goodnum-1));
    //         sig_y = sqrt(sum_y/(count_goodnum-1));
    //         sig_xp = sqrt(sum_xp/(count_goodnum-1));
    //         sig_yp = sqrt(sum_yp/(count_goodnum-1));
    //         sig_energy = sqrt(sum_w/(count_goodnum-1));
    //     }
    //     energy_spread = sig_energy / avg_w;

    //     delete [] x_h; delete [] xp_h;
    //     delete [] y_h; delete [] yp_h;
    //     delete [] phi_h; delete [] w_h; 
    //     delete [] loss_h; delete [] lloss_h;
    //     delete num_loss_h;

    //     // StopTimer(&start, &stop, "UpdateBeamParameters_CPU: ");
    //     return elapsedTime_g;
    // }


    EXPORT float UpdateBeamParameters_CPU(){
        cudaEvent_t start, stop;  
        // StartTimer(&start, &stop);

        vector<uint> loss = beam->GetLoss();
        uint num_particle = beam->num_particle;
        double* x_h = new double[num_particle];
        double* xp_h = new double[num_particle];
        double* y_h = new double[num_particle];
        double* yp_h = new double[num_particle];
        double* phi_h = new double[num_particle];
        double* w_h = new double[num_particle];
        uint* loss_h = new uint[num_particle];
        uint* lloss_h = new uint[num_particle];
        uint* num_loss_h = new uint;

        double* px_h = new double[num_particle];
        double* py_h = new double[num_particle];

        // StartTimer(&start, &stop);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        CopyBeamFromDevice(beam, x_h, xp_h, y_h, yp_h, phi_h, w_h, loss_h, lloss_h, num_loss_h);
        // StopTimer(&start, &stop, "copy beam from GPU: ");
        cudaThreadSynchronize();
        cudaEventRecord(stop, 0);   
        cudaEventSynchronize(stop);
        float elapsedTime_g;
        cudaEventElapsedTime(&elapsedTime_g, start, stop); // 計算時間差
        printf("Global memory processing time: %f (ms)\n", elapsedTime_g);

        // calculate mean
        double sum_x = 0, sum_y = 0 , sum_xp = 0, sum_yp = 0;
        double sum_xx = 0, sum_xpxp = 0, sum_xxp = 0;
        double sum_yy = 0, sum_ypyp = 0, sum_yyp = 0;
        double sum_zz = 0, sum_zpzp = 0, sum_zzp = 0;
        double sum_w2 = 0, sum_phi2 = 0, sum_phi_w = 0, sum_phi = 0;
        double sum_w = 0;
        count_goodnum = 0;
        double pzc = 0;
        double sum_px = 0, sum_py = 0, sum_px2 = 0, sum_py2 = 0, sum_px_x = 0, sum_py_y = 0;
        
        double gamma1, betagamma; 
        for(int i = 0; i < num_particle; i++){
            gamma1 = w_h[i] / beam->mass;
            betagamma = sqrt(gamma1 * (gamma1 + 2.0));
            pzc = beam->mass * betagamma;
            px_h[i] = xp_h[i] * pzc;
            py_h[i] = yp_h[i] * pzc;
            if(loss[i] == 0){
                sum_x += x_h[i];
                sum_y += y_h[i];
                sum_xp += xp_h[i];
                sum_yp += yp_h[i];
                sum_w += w_h[i];
                sum_phi += phi_h[i];

                sum_xx += x_h[i] * x_h[i];
                sum_xpxp += xp_h[i] * xp_h[i];
                sum_xxp += x_h[i] * xp_h[i];

                sum_yy += y_h[i] * y_h[i];
                sum_ypyp += yp_h[i] * yp_h[i];
                sum_yyp += y_h[i] * yp_h[i];

                sum_w2 += w_h[i] * w_h[i];
                sum_phi2 += phi_h[i] * phi_h[i];
                sum_phi_w += phi_h[i] * w_h[i];

                // sum_zz += z_h[i] * z_h[i];
                // sum_zpzp += zp_h[i] * zp_h[i];
                // sum_zzp += z_h[i] * zp_h[i];

                count_goodnum++;

                sum_px += px_h[i];
                sum_py += py_h[i];
                sum_px2 += px_h[i] * px_h[i];
                sum_py2 += py_h[i] * py_h[i];
                sum_px_x += px_h[i] * x_h[i];
                sum_py_y += py_h[i] * y_h[i];
            }
        }

        if(count_goodnum <= 2){
            avg_x = NULL;
            avg_y = NULL;
            avg_xp = NULL;
            avg_yp = NULL;
            emit_x = NULL;
            emit_y = NULL;
            sig_x = NULL;
            sig_y = NULL;
            sig_xp = NULL;
            sig_yp = NULL;
            sig_energy = NULL;
            return 0;
        }

        avg_x = sum_x / count_goodnum;
        avg_y = sum_y / count_goodnum;
        avg_xp = sum_xp / count_goodnum;
        avg_yp = sum_yp / count_goodnum;

        double avg_x2 = sum_xx / count_goodnum;
        double avg_xp2 = sum_xpxp / count_goodnum;
        double avg_xxp = sum_xxp / count_goodnum;
        double avg_y2 = sum_yy / count_goodnum;
        double avg_yp2 = sum_ypyp / count_goodnum;
        double avg_yyp = sum_yyp / count_goodnum;
        double avg_phi = sum_phi / count_goodnum;
        double avg_w2 = sum_w2 / count_goodnum;
        double avg_phi2 = sum_phi2 / count_goodnum;
        double avg_phi_w = sum_phi_w / count_goodnum;

        double avg_px = sum_px / count_goodnum;
        double avg_py = sum_py / count_goodnum;
        double avg_px2 = sum_px2 / count_goodnum;
        double avg_py2 = sum_py2 / count_goodnum;
        double avg_px_x = sum_px_x / count_goodnum;
        double avg_py_y = sum_py_y / count_goodnum;

        avg_w = sum_w / count_goodnum;

        gamma1 = avg_w / beam->mass;
        betagamma = sqrt(gamma1 * (gamma1 + 2.0));

        emit_x_trace = betagamma * sqrt((avg_x2 - avg_x * avg_x) * (avg_xp2 - avg_xp * avg_xp) - 
            (avg_xxp - avg_x * avg_xp) * (avg_xxp - avg_x * avg_xp));
        emit_y_trace = betagamma * sqrt((avg_y2 - avg_y * avg_y) * (avg_yp2 - avg_yp * avg_yp) - 
            (avg_yyp - avg_y * avg_yp) * (avg_yyp - avg_y * avg_yp));

        emit_x = sqrt((avg_x2 - avg_x * avg_x) * (avg_px2 - avg_px * avg_px) - 
            (avg_px_x - avg_x * avg_px) * (avg_px_x - avg_x * avg_px)) / beam->mass;
        emit_y = sqrt((avg_y2 - avg_y * avg_y) * (avg_py2 - avg_py * avg_py) - 
            (avg_py_y - avg_y * avg_py) * (avg_py_y - avg_y * avg_py)) / beam->mass;
        // cout<<emit_y<<","<<avg_y<<","<<avg_y2<<","<<avg_py<<","<<avg_py2<<","<<avg_py_y<<endl;


        double phi_var = avg_phi2 - avg_phi * avg_phi;
        double w_var = avg_w2 - avg_w * avg_w;
        double phi_w_covar = avg_phi_w - avg_phi * avg_w;
        emit_z = sqrt(phi_var * w_var - phi_w_covar * phi_w_covar);

        // calculate sigma
        sum_x = 0; sum_y = 0; sum_xp = 0; sum_yp = 0, sum_w = 0;
        for(int i = 0; i < num_particle; i++){
            if(loss[i] == 0){
                sum_x += (x_h[i] - avg_x)*(x_h[i] - avg_x);
                sum_y += (y_h[i] - avg_y)*(y_h[i] - avg_y);
                sum_xp += (xp_h[i] - avg_xp)*(xp_h[i] - avg_xp);
                sum_yp += (yp_h[i] - avg_yp)*(yp_h[i] - avg_yp);
                sum_w += (w_h[i] - avg_w)*(w_h[i] - avg_w);
            }
        }
        if(count_goodnum > 1){
            sig_x = sqrt(sum_x/(count_goodnum-1));
            sig_y = sqrt(sum_y/(count_goodnum-1));
            sig_xp = sqrt(sum_xp/(count_goodnum-1));
            sig_yp = sqrt(sum_yp/(count_goodnum-1));
            sig_energy = sqrt(sum_w/(count_goodnum-1));
        }
        energy_spread = sig_energy / avg_w;

        delete [] x_h; delete [] xp_h;
        delete [] y_h; delete [] yp_h;
        delete [] phi_h; delete [] w_h; 
        delete [] loss_h; delete [] lloss_h;
        delete num_loss_h;

        // StopTimer(&start, &stop, "UpdateBeamParameters_CPU: ");
        return elapsedTime_g;
    }


    EXPORT void UpdateAvgX(){
        beam->UpdateAvgX();
    }

    EXPORT double getAvgX(){
        return avg_x;
    }
    EXPORT double getAvgY(){
        return avg_y;
    }
    EXPORT double getAvgXp(){
        return avg_xp;
    }
    EXPORT double getAvgYp(){
        return avg_yp;
    }
    EXPORT double getAvgEnergy(){
        return avg_w;
    }
    EXPORT double getSigX(){
        return sig_x;
    }
    EXPORT double getSigY(){
        return sig_y;
    }
    EXPORT double getSigXp(){
        return sig_xp;
    }
    EXPORT double getSigYp(){
        return sig_yp;
    }
    EXPORT double getSigEnergy(){
        return sig_energy;
    }
    EXPORT double getEnergySpread(){
        return energy_spread;
    }
    EXPORT double getEmitX(){
        return emit_x;
    }
    EXPORT double getEmitY(){
        return emit_y;
    }
    EXPORT double getEmitZ(){
        return emit_z;
    }
    EXPORT int getGoodNum(){
        return count_goodnum;
    }




    // EXPORT void updateAll(bool r_good_only = true){
    //     beam->UpdateLongitudinalLoss();
    //     beam->UpdateLoss();
    //     beam->UpdateGoodParticleCount();
    //     beam->UpdateAvgXp();
    //     beam->UpdateAvgYp();
    //     beam->UpdateAvgX(r_good_only);
    //     beam->UpdateAvgY(r_good_only);
    //     beam->UpdateAvgPhi(r_good_only);
    //     beam->UpdateAvgRelativePhi(r_good_only);
    //     beam->UpdateAvgW(r_good_only);
    //     beam->UpdateSigXp();
    //     beam->UpdateSigYp();
    //     beam->UpdateSigX(r_good_only);
    //     beam->UpdateSigY(r_good_only);
    //     beam->UpdateSigPhi();
    //     beam->UpdateSigRelativePhi(r_good_only);
    //     beam->UpdateSigW(r_good_only);
    //     beam->UpdateEmittance();
    //     beam->UpdateStatForSpaceCharge();
    //     beam->UpdateStatForPlotting();
    //     beam->UpdateAvgXY();
    //     beam->UpdateMaxRPhi();
    //     beam->UpdateRelativePhi(r_good_only);
    // }

    // EXPORT void UpdateLongitudinalLoss(){
    //     beam->UpdateLongitudinalLoss();
    // }
    // EXPORT void UpdateLoss(){
    //     beam->UpdateLoss();
    // }
    // EXPORT void UpdateSigX(bool r_good_only = false){
    //     beam->UpdateSigX(r_good_only);
    // }
    // EXPORT void UpdateSigY(bool r_good_only = false){
    //     beam->UpdateSigY(r_good_only);
    // }


    // EXPORT double GetAvgX(bool r_good_only = false){
    //     return beam->GetAvgX(r_good_only);
    // }
    // EXPORT double GetAvgY(bool r_good_only = false){
    //     return beam->GetAvgY(r_good_only);
    // }
    // EXPORT double GetAvgPhi(bool r_good_only = false){
    //     return beam->GetAvgPhi(r_good_only);
    // }
    // EXPORT double GetAvgRelativePhi(){
    //     return beam->GetAvgRelativePhi();
    // }
    // EXPORT double GetAvgW(bool r_good_only = false){
    //     return beam->GetAvgW();
    // }
    // EXPORT double GetSigX(bool r_good_only = false){
    //     cout<<"sigx="<<beam->GetSigX(r_good_only)<<endl;
    //     return beam->GetSigX(r_good_only);
    // }
    // EXPORT double GetSigY(bool r_good_only = false){
    //     return beam->GetSigY(r_good_only);
    // }
    // EXPORT double GetSigPhi(){
    //     return beam->GetSigPhi();
    // }
    // EXPORT double GetSigRelativePhi(bool r_good_only = false){
    //     return beam->GetSigRelativePhi(r_good_only);
    // }
    // EXPORT double GetSigW(bool r_good_only = false){
    //     return beam->GetSigW(r_good_only);
    // }
    // //EXPORT double GetSigR();
    // EXPORT double GetEmittanceX(){
    //     return beam->GetEmittanceX();
    // }
    // EXPORT double GetEmittanceY(){
    //     return beam->GetEmittanceY();
    // }
    // EXPORT double GetEmittanceZ(){
    //     return beam->GetEmittanceZ();
    // }

    EXPORT void free_beam(){
        if(!beam) delete beam;
    }

    // EXPORT void init_database(char* DB_Url){
    //     db = new DBConnection(DB_Url);
    //     db->LoadLib(lib_Url);
    //     db->PrintDBs();
    // }  

    // EXPORT void init_beamline_from_DB(){
    //     beamline = BeamLine();
    //     GenerateBeamLine(beamline, db);
    //     beamline_size = beamline.GetElementNames().size();
    // }  


    EXPORT void init_Beamline(){
        beamline = BeamLine();
    }

    EXPORT void add_Drift(char* ID, double Length, double Aperture){
        Drift* drift_p = new Drift(ID);
        drift_p->SetLength( Length );
        drift_p->SetAperture( Aperture );
        beamline.AddElement(drift_p);
    }

    EXPORT void add_Bend(char* ID, double Length, double Aperture, double Angle, double AngleIn, double AngleOut, double DefaultField, double Charge, double RestEnergy){
        double Radius = Length / (Angle*RADIAN);
        double p = Charge * DefaultField * Radius * CLIGHT;
        // 能量单位：兆电子伏
        double KineticEnergy = sqrt(p * p + RestEnergy * RestEnergy) - RestEnergy;

        // cout<<Radius<<", "<<Aperture<<", "<<Angle<<", "<<AngleIn<<", "<<AngleOut<<", "<<DefaultField<<", "<<Charge<<", "<<RestEnergy<<", "<<p<<", "<<KineticEnergy<<", "<<endl;

        Dipole* dipole_p = new Dipole(ID);
        dipole_p->SetRadius( Radius );
        dipole_p->SetAngle( Angle*RADIAN );
        dipole_p->SetHalfGap( Aperture/2.0 );
        dipole_p->SetEdgeAngleIn( AngleIn*RADIAN );
        dipole_p->SetEdgeAngleOut( AngleOut*RADIAN );
        // dipole_p->SetEdgeAngleIn( -22.5*RADIAN );
        // dipole_p->SetEdgeAngleOut( -22.5*RADIAN );
        // dipole_p->SetK1( stod(one_line[5]) );
        // dipole_p->SetK2( stod(one_line[6]) );
        dipole_p->SetK1( 0.0 );
        dipole_p->SetK2( 0.0 );
        dipole_p->SetFieldIndex( 0.0 );
        dipole_p->SetAperture( Aperture );
        dipole_p->SetKineticEnergy( KineticEnergy );
        dipole_p->SetLength( Length );
        beamline.AddElement(dipole_p);
    }

    // EXPORT void add_Bend(char* ID, double Length, double Aperture){
        
    // }

    EXPORT void add_Quad(char* ID, double Length, double Aperture, double FieldGradient){
        Quad* quad_p = new Quad(ID);
        quad_p->SetLength( Length );
        quad_p->SetGradient( FieldGradient );
        quad_p->SetAperture( Aperture );
        beamline.AddElement(quad_p);
    }

    EXPORT void add_Solenoid(char* ID, double Length, double Aperture, double FieldGradient){
        Solenoid* solenoid_p = new Solenoid( ID );
        solenoid_p->SetLength( Length );
        solenoid_p->SetField( FieldGradient );
        solenoid_p->SetAperture( Aperture );
        beamline.AddElement(solenoid_p);
    }

    EXPORT void add_StraightCapillary(char* ID, double Length, double Aperture, double Current){
        StraightCapillary* StraightCapillary_p = new StraightCapillary( ID );
        StraightCapillary_p->SetLength( Length );
        StraightCapillary_p->SetAperture( Aperture );
        StraightCapillary_p->SetCurrent(Current);
        beamline.AddElement(StraightCapillary_p);
    }

    EXPORT void add_CurvedCapillary(char* ID, double Angle, double Radius, double Aperture, double Current){
        CurvedCapillary* CurvedCapillary_p = new CurvedCapillary( ID );
        CurvedCapillary_p->SetLength( abs(Angle*RADIAN * Radius) );
        CurvedCapillary_p->SetAperture( Aperture );
        CurvedCapillary_p->SetAngle( Angle*RADIAN );
        CurvedCapillary_p->SetRadius( Radius );
        CurvedCapillary_p->SetCurrent( Current );
        beamline.AddElement(CurvedCapillary_p);
    }

    EXPORT void add_ApertureRectangular(char* ID, double XLeft, double XRight, double YBottom, double YTop){
        ApertureRectangular* apertureRectangular_p = new ApertureRectangular( ID );
        apertureRectangular_p->SetIn();
        apertureRectangular_p->SetApertureXLeft( XLeft );
        apertureRectangular_p->SetApertureXRight( XRight );
        apertureRectangular_p->SetApertureYBottom( YBottom );
        apertureRectangular_p->SetApertureYTop( YTop );
        beamline.AddElement(apertureRectangular_p);
    }

    EXPORT void add_ApertureCircular(char* ID, double Aperture){
        ApertureCircular* apertureCircular_p = new ApertureCircular( ID );
        apertureCircular_p->SetIn();
        apertureCircular_p->SetAperture( Aperture );
        beamline.AddElement(apertureCircular_p);
    }


    EXPORT void load_Beamline_From_DatFile(char* filename, double ReferenceEnergy = 100){
        beamline = BeamLine();
        ifstream infile;
        infile.open(filename, ios::in);
        if (!infile.is_open())
        {
            cout << "读取文件失败" << endl;
            return;
        }
        string buf;

        vector<string> type_vector = {"DRIFT", "EDGE", "BEND", "QUAD", "SOLENOID", "ApertureCircular", "ApertureRectangular"};

        vector<vector<string> > input_dat;
        string current_type;

        // 记录每种类型元器件出现的次数
        map<string, int> device_count;
        device_count[string("DRIFT")] = 0;
        device_count[string("QUAD")] = 0;
        device_count[string("BEND")] = 0;
        device_count[string("SOLENOID")] = 0;
        device_count[string("APERTURECIRCULAR")] = 0;
        device_count[string("APERTURERECTANGULAR")] = 0;

        device_count[string("STRAIGHTCAPILLARY")] = 0;
        device_count[string("CURVEDCAPILLARY")] = 0;

        while (getline(infile,buf))
        {
            if( buf.size() == 0 || buf.at(0) == ';' || buf.at(0) == ':' ){
                continue;
            }

            vector<string> res;
            stringstream in_str(buf);
            string temp_str;
            while(in_str >> temp_str){
                res.push_back(temp_str);
            }

            string first_str = res[0];
            if( first_str.find(":") != first_str.npos){
                int split_index = first_str.find(":");
                if(first_str.size() <= split_index+1){
                    res[0] = first_str.substr(0, first_str.size()-1);
                }else{
                    res[0] = first_str.substr(split_index+1, first_str.size()-split_index-1);
                    res.insert(res.begin(), first_str.substr(0, split_index));
                }
            }else{
                transform(first_str.begin(), first_str.end(), first_str.begin(), ::toupper);
                if(device_count.count(first_str) == 1){
                    device_count[first_str] += 1;
                    res.insert(res.begin(), first_str + to_string(device_count[first_str]) );
                }else{
                    res.insert(res.begin(), first_str );
                }
            }

            current_type = res[1];
            transform(current_type.begin(), current_type.end(), current_type.begin(), ::toupper);
            res[1] = current_type;

            input_dat.push_back(res);
        }

        for(int i = 0; i < input_dat.size(); ++i){
            vector<string> one_line = input_dat[i];
            // cout<< one_line[0]<<","<< one_line[1]<<","<< one_line[2]<<","<< one_line[3]<<endl;
            current_type = one_line[1];

            // tracewin的dat文件单位为mm、T、T/m
            // 而该多粒子模拟算法单位为m、T、T/m
            if(current_type == "DRIFT"){
                Drift* drift_p = new Drift(one_line[0]);
                drift_p->SetLength( stod(one_line[2])/1000.0 );
                drift_p->SetAperture( stod(one_line[3])/1000.0 );
                beamline.AddElement(drift_p);
            }
            else if (current_type == "QUAD")
            {
                Quad* quad_p = new Quad(one_line[0]);
                quad_p->SetLength( stod(one_line[2])/1000.0 );
                quad_p->SetGradient( stod(one_line[3]) );
                quad_p->SetAperture( stod(one_line[4])/1000.0 );
                beamline.AddElement(quad_p);
            }
            else if (current_type == "EDGE")
            {
                // 在TraceWin里，EDGE-BEND-EDGE这样的组合表示偏转铁，单位需要注意
                vector<string> Bend_line = input_dat[i+1];
                vector<string> outEdge_line = input_dat[i+2];
                if(Bend_line[1] != "BEND" || outEdge_line[1] != "EDGE"){
                    cout<< "Something wrong when reading EDGE."<<endl;
                    continue;
                }
                i += 2;

                Dipole* dipole_p = new Dipole(Bend_line[0]);
                dipole_p->SetRadius( stod(Bend_line[3])/1000.0 );
                dipole_p->SetAngle( abs(stod(Bend_line[2])*RADIAN) );
                dipole_p->SetHalfGap( stod(one_line[4])/1000.0/2.0 );
                dipole_p->SetEdgeAngleIn( stod(one_line[2])*RADIAN );
                dipole_p->SetEdgeAngleOut( stod(outEdge_line[2])*RADIAN );
                // dipole_p->SetEdgeAngleIn( -22.5*RADIAN );
                // dipole_p->SetEdgeAngleOut( -22.5*RADIAN );
                // dipole_p->SetK1( stod(one_line[5]) );
                // dipole_p->SetK2( stod(one_line[6]) );
                dipole_p->SetK1( 0.0 );
                dipole_p->SetK2( 0.0 );
                dipole_p->SetFieldIndex( stod(Bend_line[4]) );
                dipole_p->SetAperture( stod(Bend_line[5])/1000.0 );
                dipole_p->SetKineticEnergy( ReferenceEnergy );
                dipole_p->SetLength( abs(dipole_p->GetRadius() * dipole_p->GetAngle()) );
                beamline.AddElement(dipole_p);

                // cout<<stod(Bend_line[3])/1000.0<<endl;
                // cout<<stod(Bend_line[2])*RADIAN<<endl;
                // cout<<stod(one_line[4])/1000.0/2.0<<endl;
                // cout<<stod(one_line[2])*RADIAN<<endl;
                // cout<<stod(outEdge_line[2])*RADIAN<<endl;
                // cout<<stod(one_line[5])<<endl;
                // cout<<stod(one_line[6])<<endl;
                // cout<<ReferenceEnergy<<endl;
                // cout<<stod(Bend_line[4])<<endl;

                // ApertureRectangular* apertureRectangular_p = new ApertureRectangular( Bend_line[0]+"-ApertureRectangular" );
                // apertureRectangular_p->SetIn();
                // apertureRectangular_p->SetApertureXLeft( stod(one_line[7])/1000.0/2.0 );
                // apertureRectangular_p->SetApertureXRight( stod(one_line[7])/1000.0/2.0 );
                // apertureRectangular_p->SetApertureYBottom( stod(one_line[4])/1000.0/2.0 );
                // apertureRectangular_p->SetApertureYTop( stod(one_line[4])/1000.0/2.0 );
                // beamline.AddElement(apertureRectangular_p);
            }
            else if (current_type == "SOLENOID")
            {
                Solenoid* solenoid_p = new Solenoid( one_line[0] );
                solenoid_p->SetLength( stod(one_line[2])/1000.0 );
                solenoid_p->SetField( stod(one_line[3]) );
                solenoid_p->SetAperture( stod(one_line[4])/1000.0 );
                beamline.AddElement(solenoid_p);
            }
            else if (current_type == "APERTURERECTANGULAR")
            {
                ApertureRectangular* apertureRectangular_p = new ApertureRectangular( one_line[0] );
                apertureRectangular_p->SetIn();
                apertureRectangular_p->SetApertureXLeft( stod(one_line[2])/1000.0 );
                apertureRectangular_p->SetApertureXRight( stod(one_line[3])/1000.0 );
                apertureRectangular_p->SetApertureYBottom( stod(one_line[4])/1000.0 );
                apertureRectangular_p->SetApertureYTop( stod(one_line[5])/1000.0 );
                beamline.AddElement(apertureRectangular_p);
            }
            else if (current_type == "APERTURECIRCULAR")
            {
                ApertureCircular* apertureCircular_p = new ApertureCircular( one_line[0] );
                apertureCircular_p->SetIn();
                apertureCircular_p->SetAperture( stod(one_line[2])/1000.0 );
                beamline.AddElement(apertureCircular_p);
            }
            else if (current_type == "STRAIGHTCAPILLARY")
            {
                StraightCapillary* StraightCapillary_p = new StraightCapillary( one_line[0]  );
                StraightCapillary_p->SetLength( stod(one_line[2])/1000.0 );
                StraightCapillary_p->SetAperture( stod(one_line[3])/1000.0 );
                StraightCapillary_p->SetCurrent(stod(one_line[4]));
                beamline.AddElement(StraightCapillary_p);
            }
            else if (current_type == "CURVEDCAPILLARY")
            {
                CurvedCapillary* CurvedCapillary_p = new CurvedCapillary( one_line[0]  );
                CurvedCapillary_p->SetLength( stod(one_line[2])*RADIAN * stod(one_line[3])/1000.0 );
                CurvedCapillary_p->SetAperture( stod(one_line[4])/1000.0  );
                CurvedCapillary_p->SetAngle( stod(one_line[2])*RADIAN );
                CurvedCapillary_p->SetRadius( stod(one_line[3])/1000.0 );
                CurvedCapillary_p->SetCurrent( stod(one_line[5])/1000.0 );
                beamline.AddElement(CurvedCapillary_p);
            }
        }

    }

    EXPORT char* get_Beamline_ElementNames(){
        delete rt_names;
        vector<string> beamlinenames = beamline.GetElementNames();
        string names_str = "";
        for(int i=0; i<beamlinenames.size(); i++){
            names_str += beamlinenames[i];
            if(i != beamlinenames.size()-1){
                names_str += ",";
            }
        }
        int charlength = strlen(names_str.c_str()) + 1;
        rt_names = new char[charlength];
        strcpy_s(rt_names, charlength, names_str.c_str());
        return rt_names;
    }

    EXPORT char* get_Beamline_ElementTypes(){
        delete rt_types;
        int bl_size = beamline.GetSize();
        string types_str = "";
        for(int i=0; i<bl_size; i++){
            types_str += beamline[i]->GetType();
            if(i != bl_size-1){
                types_str += ",";
            }
        }
        int charlength = strlen(types_str.c_str()) + 1;
        rt_types = new char[charlength];
        strcpy_s(rt_types, charlength, types_str.c_str());
        return rt_types;
    }

    EXPORT char* get_Beamline_ElementLengths(){
        delete rt_lengths;
        int bl_size = beamline.GetSize();
        string lengths_str = "";
        for(int i=0; i<bl_size; i++){
            lengths_str += to_string( beamline[i]->GetLength() );
            if(i != bl_size-1){
                lengths_str += ",";
            }
        }
        int charlength = strlen(lengths_str.c_str()) + 1;
        rt_lengths = new char[charlength];
        strcpy_s(rt_lengths, charlength, lengths_str.c_str());
        return rt_lengths;
    }

    EXPORT char* get_Beamline_ElementApertures(){
        delete rt_lengths;
        int bl_size = beamline.GetSize();
        string apertures_str = "";
        for(int i=0; i<bl_size; i++){
            apertures_str += to_string( beamline[i]->GetAperture() );
            if(i != bl_size-1){
                apertures_str += ",";
            }
        }
        int charlength = strlen(apertures_str.c_str()) + 1;
        rt_lengths = new char[charlength];
        strcpy_s(rt_lengths, charlength, apertures_str.c_str());
        return rt_lengths;
    }

    EXPORT void set_magnet_with_index(int magnet_index, double field){
        BeamLineElement* magnet = beamline[magnet_index];
        string magnet_type = magnet->GetType();
        if(magnet_type == "Dipole"){
            Dipole* dipole = dynamic_cast<Dipole*>(magnet);

            double Radius = dipole->GetRadius();
            double p =  beam->charge * field * Radius * CLIGHT;
            // 能量单位：兆电子伏
            double KineticEnergy = sqrt(p * p + beam->mass * beam->mass) - beam->mass;
            
            dipole->SetKineticEnergy(KineticEnergy);
        }else if(magnet_type == "Quad"){
            Quad* quad = dynamic_cast<Quad*>(magnet);
            quad->SetGradient(field);
        }else if(magnet_type == "Solenoid"){
            Solenoid* sole = dynamic_cast<Solenoid*>(magnet);
            sole->SetField(field);
        }else if(magnet_type == "StraightCapillary"){
            StraightCapillary* capillary = dynamic_cast<StraightCapillary*>(magnet);
            capillary->SetCurrent(field);
        }
        else if(magnet_type == "CurvedCapillary"){
            CurvedCapillary* capillary = dynamic_cast<CurvedCapillary*>(magnet);
            capillary->SetCurrent(field);
        }
    }

    EXPORT void set_magnet_with_name(char* element_name, double field){
        BeamLineElement* magnet = beamline[element_name];
        string magnet_type = magnet->GetType();
        if(magnet_type == "Dipole"){
            Dipole* dipole = dynamic_cast<Dipole*>(magnet);

            double Radius = dipole->GetRadius();
            double p =  beam->charge * field * Radius * CLIGHT;
            // 能量单位：兆电子伏
            double KineticEnergy = sqrt(p * p + beam->mass * beam->mass) - beam->mass;
            
            dipole->SetKineticEnergy(KineticEnergy);
        }else if(magnet_type == "Quad"){
            Quad* quad = dynamic_cast<Quad*>(magnet);
            quad->SetGradient(field);
        }else if(magnet_type == "Solenoid"){
            Solenoid* sole = dynamic_cast<Solenoid*>(magnet);
            sole->SetField(field);
        }else if(magnet_type == "StraightCapillary"){
            StraightCapillary* capillary = dynamic_cast<StraightCapillary*>(magnet);
            capillary->SetCurrent(field);
        }
        else if(magnet_type == "CurvedCapillary"){
            CurvedCapillary* capillary = dynamic_cast<CurvedCapillary*>(magnet);
            capillary->SetCurrent(field);
        }
    }

    EXPORT void move_magnet_with_index(int magnet_index, double delta_z){
        BeamLineElement* last_drift = beamline[magnet_index - 1];
        BeamLineElement* next_drift = beamline[magnet_index + 1];
        BeamLineElement* magnet = beamline[magnet_index];
        
        double new_length_last_drift = last_drift->GetLength() + delta_z;
        double new_length_next_drift = next_drift->GetLength() - delta_z;

        if(new_length_last_drift >= 0 && new_length_next_drift >= 0){
            last_drift->SetLength(new_length_last_drift);
            next_drift->SetLength(new_length_next_drift);
        }else{
            cout<<"The moving distance of the magnet exceeds the limit."<<endl;
        }
    }

    EXPORT void move_magnet_with_name(char* element_name, double delta_z){
        BeamLineElement* magnet = beamline[element_name];
        int magnet_index = beamline.GetElementModelIndex(element_name);
        BeamLineElement* last_drift = beamline[magnet_index - 1];
        BeamLineElement* next_drift = beamline[magnet_index + 1];
        
        double new_length_last_drift = last_drift->GetLength() + delta_z;
        double new_length_next_drift = next_drift->GetLength() - delta_z;

        if(new_length_last_drift >= 0 && new_length_next_drift >= 0){
            last_drift->SetLength(new_length_last_drift);
            next_drift->SetLength(new_length_next_drift);
        }else{
            cout<<"The moving distance of the magnet exceeds the limit."<<endl;
        }
    }

    EXPORT void init_spacecharge(uint r_nr = 32, uint r_nz = 128, int r_adj_bunch = 3){
        spacecharge = new Scheff(r_nr, r_nz, r_adj_bunch);
        spacecharge->SetInterval(0.025);
        spacecharge->SetAdjBunchCutoffW(0.8);
        spacecharge->SetRemeshThreshold(0.02);
    }  

    EXPORT void init_simulator(bool use_spacecharge){
        SetGPU(0);

        delete simulator;
        simulator = new SimulationEngine();
        PlotData* pltdata = new PlotData(particle_size);
        simulator->InitEngine(beam, &beamline, spacecharge, false, pltdata);
        if(use_spacecharge){
            simulator->SetSpaceCharge(string("on"));
        }else{
            simulator->SetSpaceCharge(string("off"));
        }
    }

    EXPORT void simulate_from_to(char* begin_element_ID, char* end_element_ID){
        // 需要在 init_simulator() 之后调用
        simulator->Simulate( begin_element_ID, end_element_ID );
    }

    EXPORT char* simulate_and_getEnvelope(bool use_spacecharge){
        cudaEvent_t start, stop;  
        StartTimer(&start, &stop);

        json rt_data;

        vector<double> position_arr, avgx_arr, avgy_arr, avgxp_arr, avgyp_arr, sigx_arr, sigy_arr, sigxp_arr, sigyp_arr, emitx_arr, emity_arr;
        vector<double> avgenergy_arr, sigenergy_arr, energyspread_arr;
        vector<int> goodnum_arr;
        vector<string> elementID_arr;

        SetGPU(0);

        delete simulator;
        simulator = new SimulationEngine();
        PlotData* pltdata = new PlotData(particle_size);
        simulator->InitEngine(beam, &beamline, spacecharge, false, pltdata);
        if(use_spacecharge){
            simulator->SetSpaceCharge(string("on"));
        }else{
            simulator->SetSpaceCharge(string("off"));
        }

        double position = 0.0;
        
        UpdateBeamParameters();
        // beam->UpdateSigX();

        position_arr.push_back(position);
        avgx_arr.push_back(getAvgX());
        avgy_arr.push_back(getAvgY());
        avgxp_arr.push_back(getAvgXp());
        avgyp_arr.push_back(getAvgYp());
        sigx_arr.push_back(getSigX());
        // sigx_arr.push_back(beam->GetSigX());
        sigy_arr.push_back(getSigY());
        sigxp_arr.push_back(getSigXp());
        sigyp_arr.push_back(getSigYp());
        emitx_arr.push_back(getEmitX());
        emity_arr.push_back(getEmitY());
        goodnum_arr.push_back(getGoodNum());
        avgenergy_arr.push_back(getAvgEnergy());
        sigenergy_arr.push_back(getSigEnergy());
        energyspread_arr.push_back(getEnergySpread());
        elementID_arr.push_back("Begin");
        
        vector<string> element_names = beamline.GetElementNames();
        for(string element: element_names){
            BeamLineElement* temp_element = beamline[element];
            position += temp_element->GetLength();
            simulator->Simulate(element, element);
            
            UpdateBeamParameters();

            position_arr.push_back(position);
            avgx_arr.push_back(getAvgX());
            avgy_arr.push_back(getAvgY());
            avgxp_arr.push_back(getAvgXp());
            avgyp_arr.push_back(getAvgYp());
            sigx_arr.push_back(getSigX());
            // sigx_arr.push_back(beam->GetSigX());
            sigy_arr.push_back(getSigY());
            sigxp_arr.push_back(getSigXp());
            sigyp_arr.push_back(getSigYp());
            emitx_arr.push_back(getEmitX());
            emity_arr.push_back(getEmitY());
            goodnum_arr.push_back(getGoodNum());
            avgenergy_arr.push_back(getAvgEnergy());
            sigenergy_arr.push_back(getSigEnergy());
            energyspread_arr.push_back(getEnergySpread());
            elementID_arr.push_back(element);
        }

        json position_json(position_arr); rt_data["z"] = position_json;
        json avgx_json(avgx_arr); rt_data["Avg_x"] = avgx_json;
        json avgy_json(avgy_arr); rt_data["Avg_y"] = avgy_json;
        json avgxp_json(avgxp_arr); rt_data["Avg_xp"] = avgxp_json;
        json avgyp_json(avgyp_arr); rt_data["Avg_yp"] = avgyp_json;
        json sigx_json(sigx_arr); rt_data["Sig_x"] = sigx_json;
        json sigy_json(sigy_arr); rt_data["Sig_y"] = sigy_json;
        json sigxp_json(sigxp_arr); rt_data["Sig_xp"] = sigxp_json;
        json sigyp_json(sigyp_arr); rt_data["Sig_yp"] = sigyp_json;
        json emitx_json(emitx_arr); rt_data["Emittance_x"] = emitx_json;
        json emity_json(emity_arr); rt_data["Emittance_y"] = emity_json;
        json goodnum_json(goodnum_arr); rt_data["Good Particle Number"] = goodnum_json;
        json avgenergy_json(avgenergy_arr); rt_data["avg_w"] = avgenergy_json;
        json sigenergy_json(sigenergy_arr); rt_data["Sig_energy"] = sigenergy_json;
        json energyspread_json(energyspread_arr); rt_data["energy_spread"] = energyspread_json;
        json elementID_json(elementID_arr); rt_data["ElementID"] = elementID_json;

        string data_string = rt_data.dump();

        int charlength = strlen(data_string.c_str()) + 1;
        delete rt_char_data;
        rt_char_data = new char[charlength];
        strcpy_s(rt_char_data, charlength, data_string.c_str());

        delete pltdata;

        StopTimer(&start, &stop, "simulate_and_getEnvelope: ");
        return rt_char_data;
    }   

    EXPORT char* simulate_and_getEnvelope_CPU(bool use_spacecharge){
        cudaEvent_t start, stop;  
        StartTimer(&start, &stop);

        json rt_data;

        vector<double> position_arr, avgx_arr, avgy_arr, avgxp_arr, avgyp_arr, sigx_arr, sigy_arr, sigxp_arr, sigyp_arr, emitx_arr, emity_arr;
        vector<double> avgenergy_arr, sigenergy_arr, energyspread_arr;
        vector<int> goodnum_arr;
        vector<string> elementID_arr;

        vector<double> emitx_trace_arr, emity_trace_arr;

        SetGPU(0);

        delete simulator;
        simulator = new SimulationEngine();
        PlotData* pltdata = new PlotData(particle_size);
        simulator->InitEngine(beam, &beamline, spacecharge, false, pltdata);
        if(use_spacecharge){
            simulator->SetSpaceCharge(string("on"));
        }else{
            simulator->SetSpaceCharge(string("off"));
        }

        double position = 0.0;
        
        UpdateBeamParameters_CPU();
        // beam->UpdateSigX();

        position_arr.push_back(position);
        avgx_arr.push_back(getAvgX());
        avgy_arr.push_back(getAvgY());
        avgxp_arr.push_back(getAvgXp());
        avgyp_arr.push_back(getAvgYp());
        sigx_arr.push_back(getSigX());
        // sigx_arr.push_back(beam->GetSigX());
        sigy_arr.push_back(getSigY());
        sigxp_arr.push_back(getSigXp());
        sigyp_arr.push_back(getSigYp());
        emitx_arr.push_back(getEmitX());
        emity_arr.push_back(getEmitY());
        goodnum_arr.push_back(getGoodNum());
        avgenergy_arr.push_back(getAvgEnergy());
        sigenergy_arr.push_back(getSigEnergy());
        energyspread_arr.push_back(getEnergySpread());
        elementID_arr.push_back("Begin");

        emitx_trace_arr.push_back(emit_x_trace);
        emity_trace_arr.push_back(emit_y_trace);
        
        vector<string> element_names = beamline.GetElementNames();
        for(string element: element_names){
            BeamLineElement* temp_element = beamline[element];
            position += temp_element->GetLength();
            simulator->Simulate(element, element);
            
            UpdateBeamParameters_CPU();

            position_arr.push_back(position);
            avgx_arr.push_back(getAvgX());
            avgy_arr.push_back(getAvgY());
            avgxp_arr.push_back(getAvgXp());
            avgyp_arr.push_back(getAvgYp());
            sigx_arr.push_back(getSigX());
            // sigx_arr.push_back(beam->GetSigX());
            sigy_arr.push_back(getSigY());
            sigxp_arr.push_back(getSigXp());
            sigyp_arr.push_back(getSigYp());
            emitx_arr.push_back(getEmitX());
            emity_arr.push_back(getEmitY());
            goodnum_arr.push_back(getGoodNum());
            avgenergy_arr.push_back(getAvgEnergy());
            sigenergy_arr.push_back(getSigEnergy());
            energyspread_arr.push_back(getEnergySpread());
            elementID_arr.push_back(element);

            emitx_trace_arr.push_back(emit_x_trace);
            emity_trace_arr.push_back(emit_y_trace);
        }

        json position_json(position_arr); rt_data["z"] = position_json;
        json avgx_json(avgx_arr); rt_data["Avg_x"] = avgx_json;
        json avgy_json(avgy_arr); rt_data["Avg_y"] = avgy_json;
        json avgxp_json(avgxp_arr); rt_data["Avg_xp"] = avgxp_json;
        json avgyp_json(avgyp_arr); rt_data["Avg_yp"] = avgyp_json;
        json sigx_json(sigx_arr); rt_data["Sig_x"] = sigx_json;
        json sigy_json(sigy_arr); rt_data["Sig_y"] = sigy_json;
        json sigxp_json(sigxp_arr); rt_data["Sig_xp"] = sigxp_json;
        json sigyp_json(sigyp_arr); rt_data["Sig_yp"] = sigyp_json;
        json emitx_json(emitx_arr); rt_data["Emittance_x"] = emitx_json;
        json emity_json(emity_arr); rt_data["Emittance_y"] = emity_json;
        json goodnum_json(goodnum_arr); rt_data["Good Particle Number"] = goodnum_json;
        json avgenergy_json(avgenergy_arr); rt_data["avg_w"] = avgenergy_json;
        json sigenergy_json(sigenergy_arr); rt_data["Sig_energy"] = sigenergy_json;
        json energyspread_json(energyspread_arr); rt_data["energy_spread"] = energyspread_json;
        json elementID_json(elementID_arr); rt_data["ElementID"] = elementID_json;

        json emitx_trace_json(emitx_trace_arr); rt_data["Emittance_x_trace"] = emitx_trace_json;
        json emity_trace_json(emity_trace_arr); rt_data["Emittance_y_trace"] = emity_trace_json;

        string data_string = rt_data.dump();

        int charlength = strlen(data_string.c_str()) + 1;
        delete rt_char_data;
        rt_char_data = new char[charlength];
        strcpy_s(rt_char_data, charlength, data_string.c_str());

        delete pltdata;

        StopTimer(&start, &stop, "simulate_and_getEnvelope_CPU: ");
        return rt_char_data;
    }   


    EXPORT void simulate_all(bool use_spacecharge = false){
        cudaEvent_t start, stop;  
        StartTimer(&start, &stop);

        SetGPU(0);

        delete simulator;
        simulator = new SimulationEngine();
        PlotData* pltdata = new PlotData(particle_size);
        simulator->InitEngine(beam, &beamline, spacecharge, false, pltdata);
        if(use_spacecharge){
            simulator->SetSpaceCharge(string("on"));
        }else{
            simulator->SetSpaceCharge(string("off"));
        }
        
        vector<string> element_names = beamline.GetElementNames();
        simulator->Simulate(element_names.front(), element_names.back());

        delete pltdata;

        StopTimer(&start, &stop, "simulate_all: ");
    }


    // EXPORT void InitSimulate(bool use_spacecharge = false, bool is_sig = true){
    //     SetGPU(0);

    //     delete simulator;
    //     simulator = new SimulationEngine();
    //     PlotData* pltdata = new PlotData(particle_size);
    //     simulator->InitEngine(beam, &beamline, spacecharge, false, pltdata);
    //     if(use_spacecharge){
    //         simulator->SetSpaceCharge(string("on"));
    //     }else{
    //         simulator->SetSpaceCharge(string("off"));
    //     }
    // }

    // EXPORT int get_envelope_size(){
    //     return envelope_size;
    // }



    // EXPORT int get_good_number(){
    //     vector<uint> transverse_loss = beam->GetLoss();
    //     int good_count = 0;
    //     for(uint state: transverse_loss){
    //         if(state == 0) good_count++;
    //     }
    //     return good_count;
    // }


    EXPORT void test(){
        beam =  new Beam(10240, 939.294, 1.0, 0.015);
        beam->InitWaterbagBeam(0, 1, 1,0, 1, 1, 0, 6.5430429, 0.00005633529, 0, 3, 500, 1);
        // beam->SaveInitialBeam();
        
        // load_Beamline_From_DatFile("F:/git_workspace/Multi-Particle-BeamLine-Simulation/Main_for_simulation/model/clapa2.dat");
        // beam1->UpdateLoss();
        // beam1->UpdateLongitudinalLoss();
        // beam1->SaveInitialBeam();
        // beam1->UpdateAvgX();
        // beam1->UpdateAvgY();
        // beam1->UpdateSigX();
        // beam1->UpdateSigY();
        // cout<<"Avgx: "<<beam1->GetAvgX()<<endl;
        // cout<<"Sigx: "<<beam1->GetSigX()<<endl;
        // cout<<"Avgy: "<<beam1->GetAvgY()<<endl;
        // cout<<"Sigy: "<<beam1->GetSigY()<<endl;

        // beam1 =  new Beam(1024000, 939.294, 1.0, 0.015);        
        // beam1->InitWaterbagBeam(0, 1, 1,0, 1, 1, 0, 6.5430429, 0.00005633529, 0, 3, 500, 1);

        // beam1->UpdateLoss();
        // beam1->UpdateLongitudinalLoss();
        // beam1->SaveInitialBeam();
        // beam1->UpdateAvgX(true);
        // beam1->UpdateAvgY(true);
        // beam1->UpdateSigX();
        // beam1->UpdateSigY();
        // cout<<"Avgx: "<<beam1->GetAvgX(true)<<endl;
        // cout<<"Sigx: "<<beam1->GetSigX(true)<<endl;
        // cout<<"Avgy: "<<beam1->GetAvgY()<<endl;
        // cout<<"Sigy: "<<beam1->GetSigY()<<endl;

        beam->UpdateAvgX();
        beam->UpdateAvgY();
        // beam->UpdateAvgXp();
        // beam->UpdateAvgYp();
        // beam->UpdateAvgW();
        // beam->UpdateSigX();
        // beam->UpdateSigY();
        // beam->UpdateSigXp();
        // beam->UpdateSigYp();
        // beam->UpdateSigW();
        // beam->UpdateEmittance();
        // beam->UpdateLoss();

        cout<<beam->GetAvgX()<<endl;
        // cout<<beam->GetSigX()<<endl;

        // avg_x = beam->GetAvgX();
        // avg_y = beam->GetAvgY();
        // avg_xp = beam->GetAvgXp();
        // avg_yp = beam->GetAvgYp();
        // emit_x = beam->GetEmittanceX();
        // emit_y = beam->GetEmittanceY();
        // emit_z = beam->GetEmittanceZ();
        // sig_x = beam->GetSigX();
        // sig_y = beam->GetSigY();
        // sig_xp = beam->GetSigXp();
        // sig_yp = beam->GetSigYp();
        // sig_energy = beam->GetSigW();

        // beam->UpdateGoodParticleCount();
        // count_goodnum = beam->GetGoodParticleNum();

        // energy_spread = sig_energy/avg_w * 2.35482005;

        // UpdateBeamParameters();

        // cout<<getAvgX()<<endl;
        // cout<<getSigX()<<endl;
        // cout<<getEmitX()<<endl;
        // cout<<getGoodNum()<<endl;
    }
}