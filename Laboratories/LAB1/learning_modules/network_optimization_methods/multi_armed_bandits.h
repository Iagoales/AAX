/* Komondor IEEE 802.11ax Simulator
 *
 * Copyright (c) 2017, Universitat Pompeu Fabra.
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007

 * Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 * Everyone is permitted to copy and distribute verbatim copies
 * of this license document, but changing it is not allowed.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the Institute nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE INSTITUTE AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE INSTITUTE OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 *
 * -----------------------------------------------------------------
 *
 * Author  : Sergio Barrachina-Muñoz and Francesc Wilhelmi
 * Created : 2016-12-05
 * Updated : $Date: 2017/03/20 10:32:36 $
 *           $Revision: 1.0 $
 *
 * -----------------------------------------------------------------
 */

 /**
 * multi_armed_bandits.h: this file contains functions related to the agents' operation
 *
 *  - This file contains the methods used by the multi-armed bandits (MAB) framework
 */

#include "../../list_of_macros.h"
#include <math.h>
#ifndef _AUX_MABS_
#define _AUX_MABS_

//new params

#define ARM_MAX 8
#define TAU_MAX 100

class MultiArmedBandit {

	// Public items
	public:

        // Print and write logs
        int save_logs;				///> Boolean for saving logs
        int print_logs;				///> Boolean for printing logs

        // General information
		int agent_id;						///> Identified of the agent using MABs
		int num_arms;						///> Number of actions
		int action_selection_strategy;		///> Index of the chosen action-selection strategy

		// Generic variables to all the learning strategies
		double initial_reward;		///> Initial reward
		int num_iterations;		///> Total number of iterations allowed

		// Arms statistics
		double *reward_per_arm;					///> Array containing the reward obtained per each arm
		double *cumulative_reward_per_arm;		///> Array containing the cumulative reward obtained per each arm
		double *average_reward_per_arm;			///> Array containing the average reward obtained per each arm
		double *estimated_reward_per_arm;		///> Array containing the estimated reward obtained per each arm
		int *times_arm_has_been_selected;		///> Array containing the times each arm is selected

		// e-greedy specific variables
		double initial_epsilon;		///> Initial epsilon parameter (exploration coefficient)
		double epsilon;				///> Epsilon parameter (exploration coefficient)

	// Methods
	public:

		/******************/
		/******************/
		/*  MAIN METHODS  */
		/******************/
		/******************/

		/**
		* Update the statistics maintained for each arm
		* @param "action_ix" [type int]: index of the action to be updated
		* @param "reward" [type double]: last reward observed from the action of interest
		*/
		void UpdateArmStatistics(int action_ix, double reward){

			if(action_ix >= 0) { // Avoid indexing errors
				// Update the reward for the chosen arm
				reward_per_arm[action_ix] = reward;
				// Update the times the chosen arm has been selected
				++times_arm_has_been_selected[action_ix];
				// Update the cumulative reward for the chosen arm
				cumulative_reward_per_arm[action_ix] += reward;
				// Update the average reward for the chosen arm
				average_reward_per_arm[action_ix] = cumulative_reward_per_arm[action_ix] /
					times_arm_has_been_selected[action_ix];
				// Update the estimated reward per arm
				//estimated_reward_per_arm[action_ix] = cumulative_reward_per_arm[action_ix] /
				//		(times_arm_has_been_selected[action_ix] + 1);
				estimated_reward_per_arm[action_ix] = ( (estimated_reward_per_arm[action_ix]*(times_arm_has_been_selected[action_ix]-1))
														+ reward ) / ( (times_arm_has_been_selected[action_ix]-1) + 2 );
				//(estimated_reward_per_arm[action_ix] + reward)
				//								/ (double)(times_arm_has_been_selected[action_ix] + 2);
			} else {
				printf("[MAB] ERROR: The action ix (%d) is not correct!\n", action_ix);
				exit(EXIT_FAILURE);
			}
		}

		/**
		* Select a new action according to the chosen action selection strategy
		* @return "action_ix" [type int]: index of the selected action
		*/
		int SelectNewAction(int *available_arms, int current_arm) {
			int arm_ix;
			// Select an action according to the chosen strategy
			switch(action_selection_strategy) {
				/*
				 * epsilon-greedy strategy:
				 */
				case STRATEGY_EGREEDY:{
					// Update epsilon
					epsilon = initial_epsilon / sqrt( (double) num_iterations);
					// Pick an action according to e-greedy
                    arm_ix = PickArmEgreedy(num_arms, average_reward_per_arm, epsilon, available_arms);
                    //printf("Action selected = %d\n", arm_ix);
					break;
				}
				/*
				 * Thompson sampling strategy:
				 */
				case STRATEGY_THOMPSON_SAMPLING:{
					// Pick an action according to Thompson sampling
                    arm_ix = PickArmThompsonSampling(num_arms,
						estimated_reward_per_arm, times_arm_has_been_selected, available_arms);
					break;
				}
				/*
				 * Upper Confidence Bound (UCB):
				 */
				case STRATEGY_UCB:{
					// Pick an action according to Thompson sampling
					arm_ix = PickArmUCB(num_arms, average_reward_per_arm,
							times_arm_has_been_selected, available_arms, num_iterations);
					break;
				}
                /*
                 * Sequential selection strategy:
                 */
                case STRATEGY_SEQUENTIAL:{
                    // Pick an action according to Thompson sampling
                    arm_ix = PickArmSequentially(num_arms, available_arms, current_arm);
                    break;
                }
                /*
                 * ML4Net strategy:
                 */
                case STRATEGY_ML4Net:{
						arm_ix = PickArmML4Net(num_arms,
						average_reward_per_arm,
						times_arm_has_been_selected,
						available_arms,
						num_iterations);
					break;
				}

				default:{
					printf("[MAB] ERROR: '%d' is not a correct action-selection strategy!\n", action_selection_strategy);
					PrintAvailableActionSelectionStrategies();
					exit(EXIT_FAILURE);
				}
			}
            // Increase the number of iterations
            ++ num_iterations;
            // Return the selected action
			return arm_ix;
		}

		/****************************/
		/****************************/
		/*  EPSILON-GREEDY METHODS  */
		/****************************/
		/****************************/

		/**
		 * Select an action according to the epsilon-greedy strategy
		 * @param "num_arms" [type int]: number of possible actions
		 * @param "reward_per_arm" [type double]: array containing the last stored reward for each action
		 * @param "epsilon" [type double]: current exploration coefficient
		 * @return "action_ix" [type int]: index of the selected action
		 */
		int PickArmEgreedy(int num_arms, double *reward_per_arm, double epsilon, int *available_arms) {

			double rand_number = ((double) rand() / (double)RAND_MAX);
			int action_ix;

			if (rand_number < epsilon) { //EXPLORE
				action_ix = rand() % num_arms;
				int counter(0);
				while (!available_arms[action_ix]) {
					action_ix = rand() % num_arms;
					if(counter > 1000) break; // To avoid getting stuck (none of the actions is available)
				}
				//printf("EXPLORE: Selected action %d (available = %d), reward = %f\n", action_ix, available_arms[action_ix], reward_per_arm[action_ix]);
			} else { //EXPLOIT
				double max = 0;
				for (int i = 0; i < num_arms; i ++) {
					//printf("   - reward_per_arm[%d] = %f\n", i, reward_per_arm[i]);
					if(available_arms[i] && reward_per_arm[i] >= max) {
						max = reward_per_arm[i];
						action_ix = i;
					}
				}
				//printf("EXPLOIT: Selected action %d (available = %d), reward = %f\n", action_ix, available_arms[action_ix], reward_per_arm[action_ix]);
			}
			return action_ix;

		}

		/*******************************/
		/*******************************/
		/*  THOMPSON SAMPLING METHODS  */
		/*******************************/
		/*******************************/

		double rand_normal() {
		  double u1 = (double)rand() / RAND_MAX;
		  double u2 = (double)rand() / RAND_MAX;
		  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
		}

		double sample_rand_normal(double mu, double sigma) {
		  return mu + sigma * rand_normal();
		}

		/**
		 * Select an action according to the Thompson sampling strategy
		 * @param "num_arms" [type int]: number of possible actions
		 * @param "estimated_reward_per_arm" [type double*]: array containing the estimated reward for each action
		 * @param "times_arm_has_been_selected" [type int*]: array containing the times each action has been selected
		 * @return "action_ix" [type int]: index of the selected action
		 */
		int PickArmThompsonSampling(int num_arms, double *estimated_reward_per_arm,
			int *times_arm_has_been_selected, int *available_arms) {
			//TODO: validate the behavior of this implementation
			int action_ix(-1);
			double *theta = new double[num_arms];
			double std;
			int KMAX(1);
			// Compute the posterior probability of each arm
			for (int i = 0; i < num_arms; ++i) {
				if (available_arms[i]) {
					std = 1.0/(double)(1+times_arm_has_been_selected[i]);
					theta[i] = 0;
					for (int k = 0; k < KMAX; ++k){
						theta[i] += sample_rand_normal(estimated_reward_per_arm[i], std);
					}
				}
			}
			// Find the action with the highest likelihood
			double max = -10000;
			for (int i = 0; i < num_arms; ++i) {
				if(theta[i] > max && available_arms[i]) {
					max = theta[i];
					action_ix = i;
				}
				//  TODO: elseif(theta[i] == max) --> Break ties!
			}
//			printf("Selected action %d (available = %d)\n", action_ix, available_arms[action_ix]);
			return action_ix;
		}

		/**
		 * Select an action according to the UCB sampling strategy
		 * @param "num_arms" [type int]: number of possible actions
		 * @param "estimated_reward_per_arm" [type double*]: array containing the estimated reward for each action
		 * @param "times_arm_has_been_selected" [type int*]: array containing the times each action has been selected
		 * @return "action_ix" [type int]: index of the selected action
		 */
		int PickArmUCB(int num_arms, double *average_reward_per_arm,
			int *times_arm_has_been_selected, int *available_arms, int num_iterations) {
			//TODO: validate the behavior of this implementation
			int action_ix(-1);
			double *ucb_estimate = new double[num_arms];
			double max = -10000;
			// Compute the posterior probability of each arm
			for (int i = 0; i < num_arms; ++i) {
				if (available_arms[i]) {
					ucb_estimate[i] = average_reward_per_arm[i] +
							sqrt((2*log(num_iterations))/times_arm_has_been_selected[i]);
					if (ucb_estimate[i] > max) {
						max = ucb_estimate[i];
						action_ix = i;
					}
				}
			}
//			printf("Selected action %d (available = %d)\n", action_ix, available_arms[action_ix]);
			return action_ix;
		}


		/*******************/
        /*******************/
        /*  ML4NET METHODS */
        /*******************/
        /*******************/

        /**
         * Select an action according to the Thompson sampling strategy
         * @param "num_arms" [type int]: number of possible actions
         * @param TO BE COMPLETED
         * ...
         * @return "action_ix" [type int]: index of the selected action
         */
		/*  util: N(0,1) con Box–Muller  ------------------------------ */
		static double randn(void)
		{
			double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
			double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
			return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
		}
		

		int PickArmML4Net(int   num_arms,
						  double *average_reward_per_arm,
						  int    *times_arm_has_been_selected,
						  int    *available_arms,
						  int     num_iterations)
		{
			/* ---------- parámetres necessaris ---------- */
			const double tau0        = 0.65;   //tau inicial
			const double eps         = 0.005;  // mini-ε exploración
			const double drop_thresh = 0.075;   // 10 % de caída 

			/* ---------- descenso de temperatura ---------- */
			double tau = tau0 / sqrt((double)(num_iterations > 0 ? num_iterations : 1));

			/* ---------- elección “guess” por máxima recompensa ---------- */
			int    best_arg = -1;
			double best_avg = -1e300;
			for (int a = 0; a < num_arms; ++a) {
				if (!available_arms[a]) continue;
				if (average_reward_per_arm[a] > best_avg) {
					best_avg = average_reward_per_arm[a];
					best_arg = a;
				}
			}

			/* ---------- detección de caída brusca ---------- */
			{
				static double prev_best = -1.0;
				if (prev_best >= 0.0) {
					double curr = average_reward_per_arm[best_arg];
					if (curr < prev_best * (1.0 - drop_thresh)) {
						/* forzar exploración en otro brazo */
						int alt;
						do { alt = rand() % num_arms; }
						while (!available_arms[alt] || alt == best_arg);
						prev_best = curr;
						return alt;
					}
				}
				prev_best = average_reward_per_arm[best_arg];
			}

			/* ---------- mini-ε exploración suave ---------- */
			if ((double)rand() / (double)RAND_MAX < eps) {
				int alt;
				do { alt = rand() % num_arms; }
				while (!available_arms[alt]);
				return alt;
			}

			/* ---------- Soft-max sampling ---------- */
			double sum_exp = 0.0;
			static double prob[ARM_MAX];
			for (int a = 0; a < num_arms; ++a) {
				if (!available_arms[a]) {
					prob[a] = 0.0;
				} else {
					prob[a] = exp(average_reward_per_arm[a] / tau);
					sum_exp += prob[a];
				}
			}
			if (sum_exp <= 0.0) {
				/* fallback uniforme si algo raro pasa */
				for (int a = 0; a < num_arms; ++a)
					if (available_arms[a]) prob[a] = 1.0;
				sum_exp = 0.0;
				for (int a = 0; a < num_arms; ++a)
					sum_exp += prob[a];
			}
			for (int a = 0; a < num_arms; ++a)
				prob[a] /= sum_exp;

			double r = (double)rand() / (double)RAND_MAX;
			double acc = 0.0;
			for (int a = 0; a < num_arms; ++a) {
				acc += prob[a];
				if (r <= acc && available_arms[a]) {
					return a;
				}
			}

			/* fallback final */
			for (int a = 0; a < num_arms; ++a)
				if (available_arms[a]) return a;
			return 0;
		}
		
		
		/**

		/*  PickArmML4Net – Soft-max (τ↓) + ping normal únic 
		int PickArmML4Net(int   num_arms,
						  double *average_reward_per_arm,
						  int    *times_arm_has_been_selected,
						  int    *available_arms,
						  int     num_iterations)
		{
			// ---------- parámetros Soft-max ----------
			const double tau0 = 0.65;       

			//
			double tau = tau0 / sqrt((double)(num_iterations > 0 ? num_iterations : 1));

			// ---------- variables estáticas de
			static int ping_set = 0;           
			static int t_ping   = 0;            

			// (1) Sortea el ping en la primera llamada
				   
			if (!ping_set && num_iterations == 0) {
				int  t_max  = 40;                               
				double mu   = 0.5  * t_max;                     
				double sigma= t_max / 6.0;                    
				int cand;
				do {
					cand = (int)(mu + sigma * randn());
				} while (cand < 1 || cand > t_max);
				t_ping   = cand;
				ping_set = 1;
			}

			// (2) Momento de exploración forzada? 
			if (ping_set && num_iterations == t_ping) {
				// elige un brazo disponible al azar
				int alt;
				do { alt = rand() % num_arms; }
				while (!available_arms[alt]);
				return alt;
			}

			//
			double prob[ARM_MAX] = {0.0};
			double sum_exp = 0.0;

			for (int a = 0; a < num_arms; ++a) {
				if (!available_arms[a]) continue;
				prob[a]  = exp( average_reward_per_arm[a] / tau );
				sum_exp += prob[a];
			}
			if (sum_exp == 0.0) {                           
				for (int a = 0; a < num_arms; ++a)
					if (available_arms[a]) { prob[a] = 1.0; sum_exp += 1.0; }
			}
			for (int a = 0; a < num_arms; ++a) prob[a] /= sum_exp;

			
			double r   = (double)rand() / (double)RAND_MAX;
			double acc = 0.0;
			for (int a = 0; a < num_arms; ++a) {
				acc += prob[a];
				if (r <= acc) return a;
			}
			//
			for (int a = 0; a < num_arms; ++a)
				if (available_arms[a]) return a;
			return 0;
		}
		
		**/

		
		/*  PickArmML4Net – Variable Sliding-Window UCB1 - UPC Paper (Descartat perq es NON-SATATIONARY)*/
		/**
		
		int PickArmML4Net(int num_arms,
						  double *average_reward_per_arm,
						  int *times_arm_has_been_selected,
						  int *available_arms,
						  double *reward_per_arm,
						  int   num_iterations)
		{
			//---------- parámetros fijos ---------- 
			const int tau_min  = 5;
			const int tau_max  = 100;
			const int tau_step = 5;
			const int m1       = 10;   /
			const int m0       = 5;    

			
			static int    tau = 20;                
			static int    same_sign = 0, alt_sign = 0;
			static double prev_error = 0.0;

			//buffers circulares   [brazo][0..TAU_MAX-1] 
			static double buffer[ARM_MAX][TAU_MAX] = {{0.0}};
			static int    pos   [ARM_MAX]          = {0};

			//---------- 1. añadir recompensa del paso anterior ---------- 
			int last_arm = -1;
			for (int a = 0; a < num_arms; ++a)
				if (reward_per_arm[a] >= 0.0) { last_arm = a; break; } // Komondor pone −1 si no jugó 

			if (last_arm >= 0) {
				buffer[last_arm][ pos[last_arm] ] = reward_per_arm[last_arm];
				pos[last_arm] = (pos[last_arm] + 1) % TAU_MAX;
			}

			// ---------- 2. media reciente y contador en ventana τ ---------- 
			double est_reward[ARM_MAX] = {0.0};
			int    count      [ARM_MAX] = {0};

			for (int a = 0; a < num_arms; ++a) {
				if (!available_arms[a]) continue;

				int p = pos[a];
				for (int k = 0; k < tau; ++k) {
					int idx = (p - 1 - k + TAU_MAX) % TAU_MAX;
					double r = buffer[a][idx];
					if (r == 0.0 && k >= times_arm_has_been_selected[a]) break; // no hay muestra 
					est_reward[a] += r;
					count[a]++;
				}
				if (count[a] > 0)
					est_reward[a] /= (double)count[a];
			}

			//---------- 3. UCB en esa ventana ---------- 
			double best_ucb = -1.0;
			int    best_arm = 0;

			int t = (num_iterations > 1) ? num_iterations : 1;   /

			for (int a = 0; a < num_arms; ++a) {
				if (!available_arms[a]) continue;

				int    Ni = (count[a] > 0) ? count[a] : 1;
				double mu = est_reward[a];
				double ucb = mu + sqrt( 2.0 * log( (double)t ) / (double)Ni );

				if (ucb > best_ucb) { best_ucb = ucb; best_arm = a; }
			}

			//---------- 4. ajustar τ según el signo del error ---------- 
			double err = 0.0;
			if (last_arm >= 0)
				err = reward_per_arm[last_arm] - est_reward[last_arm];

			if (err * prev_error > 0.0) {                 //mismo signo 
				same_sign++;  alt_sign = 0;
				if (same_sign >= m1 && tau > tau_min) {
					tau = (tau - tau_step > tau_min) ? tau - tau_step : tau_min;
					same_sign = alt_sign = 0;
				}
			}
			else if (err * prev_error < 0.0) {            //signo alterno 
				alt_sign++;   same_sign = 0;
				if (alt_sign >= m0 && tau < tau_max) {
					tau = (tau + tau_step < tau_max) ? tau + tau_step : tau_max;
					same_sign = alt_sign = 0;
				}
			}
			prev_error = err;

			return best_arm;
		}
		**/
		

        /*******************/
        /*******************/
        /*  OTHER METHODS  */
        /*******************/
        /*******************/


        /**
         * Select an action according to the PickArmSequentially strategy
         * @param "num_arms" [type int]: number of possible actions
         * @param "available_arms" [type int*]: array with the available arms
         * @return "current_arm_ix" [type int]: index of the selected action
         */
        int PickArmSequentially(int num_arms, int *available_arms, int current_arm_ix) {
            int arm_ix(1);
            if (num_iterations == 1) {
                arm_ix = num_iterations;
            } else {
                arm_ix = (current_arm_ix + 1) % num_arms;
            }
            if (available_arms[arm_ix] != 1) {
                arm_ix = PickArmSequentially(num_arms, available_arms, arm_ix);
            }
            return arm_ix;
        }

		/*************************/
		/*************************/
		/*  PRINT/WRITE METHODS  */
		/*************************/
		/*************************/

		/**
		* Print or write the statistics of each arm
		* @param "write_or_print" [type int]: variable to indicate whether to print on the  console or to write on the the output logs file
		* @param "logger" [type Logger]: logger object to write on the output file
		* @param "sim_time" [type double]: simulation time
		*/
		void PrintOrWriteStatistics(int write_or_print, Logger &logger, double sim_time) {
			// Write or print according the input parameter "write_or_print"
			switch(write_or_print){
				// Print logs in console
				case PRINT_LOG:{
					if(print_logs){
						printf("%s Reward per arm: ", LOG_LVL3);
						for(int n = 0; n < num_arms; n++){
							printf("%f  ", reward_per_arm[n]);
						}
						printf("\n%s Cumulative reward per arm: ", LOG_LVL3);
						for(int n = 0; n < num_arms; n++){
							printf("%f  ", cumulative_reward_per_arm[n]);
						}
						printf("\n%s Times each arm has been selected: ", LOG_LVL3);
						for(int n = 0; n < num_arms; n++){
							printf("%d  ", times_arm_has_been_selected[n]);
						}
						printf("\n%s Estimated reward per arm: ", LOG_LVL3);
						for(int n = 0; n < num_arms; n++){
							printf("%f  ", estimated_reward_per_arm[n]);
						}
						printf("\n");
					}
					break;
				}
				// Write logs in agent's output file
				case WRITE_LOG:{
					if(save_logs) fprintf(logger.file, "%.15f;A%d;%s;%s Reward per arm: ",
						sim_time, agent_id, LOG_C00, LOG_LVL3);
					for(int n = 0; n < num_arms; n++){
						 if(save_logs){
							 fprintf(logger.file, "%f  ", reward_per_arm[n]);
						 }
					}
					if(save_logs) fprintf(logger.file, "\n%.15f;A%d;%s;%s Cumulative reward per arm: ",
						sim_time, agent_id, LOG_C00, LOG_LVL3);
					for(int n = 0; n < num_arms; n++){
						 if(save_logs){
							 fprintf(logger.file, "%f  ", cumulative_reward_per_arm[n]);
						 }
					}
					fprintf(logger.file, "\n%.15f;A%d;%s;%s Times each arm has been selected: ",
									sim_time, agent_id, LOG_C00, LOG_LVL3);
					for(int n = 0; n < num_arms; n++){
						if(save_logs){
							fprintf(logger.file, "%d ", times_arm_has_been_selected[n]);
						}
					}
					if(save_logs) fprintf(logger.file, "\n%.15f;A%d;%s;%s Estimated reward per arm: ",
						sim_time, agent_id, LOG_C00, LOG_LVL3);
					for(int n = 0; n < num_arms; n++){
						 if(save_logs){
							 fprintf(logger.file, "%f  ", estimated_reward_per_arm[n]);
						 }
					}
					if(save_logs) fprintf(logger.file, "\n");
					break;
				}
			}
		}

		/**
		 * Print the available ML mechanisms types
		 */
		void PrintAvailableActionSelectionStrategies(){
			printf("%s Available types of action-selection strategies:\n", LOG_LVL2);
			printf("%s STRATEGY_EGREEDY (%d)\n", LOG_LVL3, STRATEGY_EGREEDY);
			printf("%s STRATEGY_THOMPSON_SAMPLING (%d)\n", LOG_LVL3, STRATEGY_THOMPSON_SAMPLING);
			printf("%s STRATEGY_UCB (%d)\n", LOG_LVL3, STRATEGY_UCB);
			printf("%s STRATEGY_SEQUENTIAL (%d)\n", LOG_LVL3, STRATEGY_SEQUENTIAL);		
			printf("%s STRATEGY_ML4Net (%d)\n", LOG_LVL3, STRATEGY_ML4Net);
		}

		/***********************/
		/***********************/
		/*  AUXILIARY METHODS  */
		/***********************/
		/***********************/

		/**
		 * Initialize the variables used by the Bandits framework
		 */
		void InitializeVariables(){
			// TODO: generate file that stores algorithm-specific variables
			initial_epsilon = 1;
			epsilon = initial_epsilon;
			initial_reward = 0; //(double) 1/num_arms;
			num_iterations = 1;
			// Initialize the rewards assigned to each arm
			reward_per_arm = new double[num_arms];
			cumulative_reward_per_arm = new double[num_arms];
			average_reward_per_arm = new double[num_arms];
			estimated_reward_per_arm = new double[num_arms];
			// Initialize the array containing the times each arm has been played
			times_arm_has_been_selected = new int[num_arms];
			for(int i = 0; i < num_arms; ++i){
				reward_per_arm[i] = initial_reward;	// Set the initial reward
				cumulative_reward_per_arm[i] = initial_reward;
				average_reward_per_arm[i] = initial_reward;
				estimated_reward_per_arm[i] = initial_reward;
				times_arm_has_been_selected[i] = 0;
			}
		}

};

#endif
