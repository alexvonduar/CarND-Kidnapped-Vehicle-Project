/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (!is_initialized) {
		default_random_engine gen;

		normal_distribution<double> dist_x(x, std[0]);
		normal_distribution<double> dist_y(y, std[1]);
		normal_distribution<double> dist_psi(theta, std[2]);

		num_particles = 1000;

		for (int i = 0; i < num_particles; ++i) {
			Particle particle;
			particle.x = dist_x(gen);
			particle.y = dist_y(gen);
			particle.theta = dist_psi(gen);
			particle.weight = 1;
			particles.push_back(particle);
			weights.push_back(1);
		}
		is_initialized = true;

	} else {
		printf("already initialized!\n");
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;
	for (int i = 0; i < particles.size(); ++i)
	{
		double x;
		double y;

		if (fabs(yaw_rate) < numeric_limits<double>::epsilon())
		{
			x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
			y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
		}
		else
		{
			x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
			y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
		}

		double theta = yaw_rate * delta_t + particles[i].theta;

		normal_distribution<double> dist_x(x, std_pos[0]);
		normal_distribution<double> dist_y(y, std_pos[1]);
		normal_distribution<double> dist_psi(theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_psi(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int i = 0; i < observations.size(); ++i)
	{
		double min = numeric_limits<double>::infinity();
		for (int j = 0; j < predicted.size(); ++j)
		{
			double d = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (d < min)
			{
				min = d;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

static const double trans_x(const double& x_obs,
	                        const double& y_obs,
	                        const double& x_part,
	                        const double& theta)
{
	double rslt = x_part + x_obs * cos(theta) - y_obs * sin(theta);
	return rslt;
}

static const double trans_y(const double& x_obs,
							const double& y_obs,
							const double& y_part,
							const double& theta)
{
	double rslt = y_part + x_obs * sin(theta) + y_obs * cos(theta);
	return rslt;
}

static const double multivar_gaussian(const double& x,
									  const double& mu_x,
									  const double& sig_x,
									  const double& y,
									  const double& mu_y,
									  const double& sig_y)
{
	// calculate normalization term
	double gauss_norm = (double)1.0 / (2 * M_PI * sig_x * sig_y);


	// calculate exponent
	double x_exponent = ((x - mu_x) * (x - mu_x)) / (2 * sig_x * sig_x);
	double y_exponent = ((y - mu_y) * (y - mu_y)) / (2 * sig_y * sig_y);

	// calculate weight using normalization terms and exponent
	double weight = gauss_norm * exp(-(x_exponent + y_exponent));
	return weight;
}

static int find(int id, const vector<LandmarkObs>& obs)
{
	int i;
	for (i = 0; i < obs.size(); ++i)
	{
		if (id == obs[i].id)
		{
			break;
		}
	}

	return i;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < particles.size(); ++i)
	{
		vector<LandmarkObs> transObs;

		for (int j = 0; j < observations.size(); ++j)
		{
			LandmarkObs transOb;
			transOb.x = trans_x(observations[j].x, observations[j].y, particles[i].x, particles[i].theta);
			transOb.y = trans_y(observations[j].x, observations[j].y, particles[i].y, particles[i].theta);

			transObs.push_back(transOb);
		}

		vector<LandmarkObs> predictedObs;

		for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
			double d = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[i].x, particles[i].y);
			if (d <= sensor_range)
			{
				predictedObs.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
			}
		}

		if (predictedObs.size() > 0)
		{

			dataAssociation(predictedObs, transObs);

			particles[i].weight = 1.0; //reset the weight of the particle
			for (int j = 0; j < transObs.size(); ++j)
			{
				int idx = find(transObs[j].id, predictedObs);


				particles[i].weight *= multivar_gaussian(transObs[j].x,
					                                     predictedObs[idx].x,
														 std_landmark[0],
														 transObs[j].y,
														 predictedObs[idx].y,
														 std_landmark[1]);
			}
			weights[i] = particles[i].weight;
		}
		else
		{
			weights[i] = 0.0;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<> d(weights.begin(), weights.end());
	vector<Particle> p_res;

	for (int n = 0; n < particles.size(); ++n)
	{
		p_res.push_back(particles[d(gen)]);
	}

	particles = p_res;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
