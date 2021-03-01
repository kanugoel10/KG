/*
 *    AccuracyUpdatedEnsemble2.java
 *    Copyright (C) 2010 Poznan University of Technology, Poznan, Poland
 *    @author Dariusz Brzezinski (dariusz.brzezinski@cs.put.poznan.pl)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package moa.classifiers.meta;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import moa.classifiers.Classifier;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.BoastedInstance;
import moa.core.BoastedInstances;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * <code>DADDE</code> This file contains dual ensemble both updated by weighing
 * function which considers last chunk MSI as well and assigns MSR to perfect
 * chunk
 */
public class DADDE extends IncrementalAWEDetectorAbstain {

	private static final long serialVersionUID = 1L;

	public FloatOption lamdaOption = new FloatOption("lamda1", 'j', "The lambda options.", 2.0, 0.0, Float.MAX_VALUE);

	public FloatOption lamdaWarnOption = new FloatOption("lamda2", 'h', "Lamda 2 value.", 2, 0.0, Float.MAX_VALUE);

	public IntOption maxByteSizeOption = new IntOption("maxByteSize", 'm', "Maximum memory consumed by ensemble.",
			33554432, 0, Integer.MAX_VALUE);

	/**
	 * The weights of stored classifiers. weights[x][0] = weight weights[x][1] =
	 * classifier number in learners
	 */
	protected double[][] weights1;
	protected double[][] msi;
	protected double[][] msi2;

	/**
	 * The weights of stored classifiers. weights[x][0] = weight weights[x][1] =
	 * classifier number in learners
	 */
	protected double[][] weights2;

	/**
	 * Ensemble classifiers.
	 */
	protected Classifier[] learners1;

	/**
	 * Ensemble classifiers.
	 */
	protected Classifier[] learners2;

	/**
	 * Candidate classifier.
	 */
	protected Classifier candidate;

	/**
	 * Current chunk of instances.
	 */
	// protected Instances currentChunk;

	@Override
	public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();

		super.prepareForUseImpl(monitor, repository);
	}

	@Override
	public void resetLearningImpl() {
		super.resetLearningImpl();
		this.currentChunk = null;
		this.warningChunk = null;// ************VPN
		this.classDistributions = null;
		this.classDistributionsForWarrning = null;
		this.processedInstances = 0;
		this.learners1 = new Classifier[0];
		this.learners2 = new Classifier[0];// ****************VPN

		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();
	}

	/**
	 * Determines whether the classifier is randomizable.
	 */
	public boolean isRandomizable() {
		return true;
	}

	/**
	 * Predicts a class for an example.
	 */

	public double[] getVotesForInstance(Instance inst) {
		DoubleVector globalPredictionForEnsemble2 = new DoubleVector();
		DoubleVector globalPredictionForEnsemble1 = new DoubleVector();
		DoubleVector combinedglobalPrediction = new DoubleVector();
		if (this.trainingWeightSeenByModel > 0.0) {
			if (this.useBothEnsemble) {
				getCombinedGlobalPredictions(inst, combinedglobalPrediction);
				return combinedglobalPrediction.getArrayRef();
			} else if (this.useEnsemble1) {
				getGlobalPredictionForEnsemble1(inst, globalPredictionForEnsemble1);

				return globalPredictionForEnsemble1.getArrayRef();
			} else {
				getGlobalPredictionForEnsemble2(inst, globalPredictionForEnsemble2);

				return globalPredictionForEnsemble2.getArrayRef();
			}
		}

		return globalPredictionForEnsemble1.getArrayRef();
	}

	private void getGlobalPredictionForEnsemble2(Instance inst, DoubleVector globalPredictionForEnsemble2) {
		for (int i = 0; i < this.learners2.length; i++) {
			if (this.weights2[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners2[(int) this.weights2[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights2[i][0] / (1.0 * this.learners2.length + 1.0));
					globalPredictionForEnsemble2.addValues(vote);
				}
			}
		}
	}

	private void getGlobalPredictionForEnsemble1(Instance inst, DoubleVector globalPredictionForEnsemble1) {
		for (int i = 0; i < this.learners1.length; i++) {
			if (this.weights1[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners1[(int) this.weights1[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights1[i][0] / (1.0 * this.learners1.length + 1.0));
					globalPredictionForEnsemble1.addValues(vote);
				}
			}
		}
	}

	private void getCombinedGlobalPredictions(Instance inst, DoubleVector combinedGlobalPrediction) {
		for (int i = 0; i < this.learners1.length; i++) {
			if (this.weights1[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners1[(int) this.weights1[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights1[i][0] / (1.0 * this.learners1.length + 1.0));
					combinedGlobalPrediction.addValues(vote);
				}
			}
		}

		for (int i = 0; i < this.learners2.length; i++) {
			if (this.weights2[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners2[(int) this.weights2[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights2[i][0] / (1.0 * this.learners2.length + 1.0));
					combinedGlobalPrediction.addValues(vote);
				}
			}
		}
	}

	private double[] getGlobalPredictionForEnsemble1(Instance inst) {
		DoubleVector globalPredictionForEnsemble1 = new DoubleVector();
		for (int i = 0; i < this.learners1.length; i++) {
			if (this.weights1[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners1[(int) this.weights1[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights1[i][0] / (1.0 * this.learners1.length + 1.0));
					globalPredictionForEnsemble1.addValues(vote);
				}

				return globalPredictionForEnsemble1.getArrayRef();
			}
		}
		return globalPredictionForEnsemble1.getArrayRef();
	}

	private double[] getGlobalPredictionForEnsemble2(Instance inst) {
		DoubleVector globalPredictionForEnsemble2 = new DoubleVector();
		for (int i = 0; i < this.learners2.length; i++) {
			if (this.weights2[i][0] > 0.0) {
				DoubleVector vote = new DoubleVector(
						this.learners2[(int) this.weights2[i][1]].getVotesForInstance(inst));

				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					// scale weight and prevent overflow
					vote.scaleValues(this.weights2[i][0] / (1.0 * this.learners2.length + 1.0));
					globalPredictionForEnsemble2.addValues(vote);
				}

				return globalPredictionForEnsemble2.getArrayRef();
			}
		}
		return globalPredictionForEnsemble2.getArrayRef();
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

	@Override
	public Classifier[] getSubClassifiers() {
		return this.learners1.clone();
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// Made a call to the parent that is detector
		super.trainOnInstanceImpl(inst);
		ExecutorService executorServiceForGlobalPrediction = Executors.newFixedThreadPool(2);
		executorServiceForGlobalPrediction.execute(new Runnable() {
			// after training done
			@Override
			public void run() {
				if (Utils.maxIndex(getGlobalPredictionForEnsemble1(inst)) == inst.classValue()) {
					correctCount1++;
				}
			}
		});
		executorServiceForGlobalPrediction.execute(new Runnable() {
			@Override
			public void run() {
				if (Utils.maxIndex(getGlobalPredictionForEnsemble2(inst)) == inst.classValue()) {
					correctCount2++;
				}
			}
		});

		executorServiceForGlobalPrediction.shutdown();
		try {
			executorServiceForGlobalPrediction.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {

		}

	}

	/**
	 * Processes a chunk of instances. This method is called after collecting a
	 * chunk of examples.
	 */
	protected void processChunk(boolean isDrift, boolean isReplace) {
		ExecutorService executorServiceForEnsembleUpdationAndWeightUpdation = Executors.newFixedThreadPool(2);
		processForEnsemble2(executorServiceForEnsembleUpdationAndWeightUpdation);
		if (!isDrift && (isReplace || learners1.length < this.memberCountOption.getValue())) {
			processForEnsemble1(executorServiceForEnsembleUpdationAndWeightUpdation);
		} // ENsemble 1 is passive
		executorServiceForEnsembleUpdationAndWeightUpdation.shutdown();
		try {
			executorServiceForEnsembleUpdationAndWeightUpdation.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {

		}

		trainingWithBagging(isDrift);

		this.classDistributions = null;
		this.classDistributionsForWarrning = null;
		if (!isDrift) {
			this.currentChunk = null;
		}
		this.warningChunk = null;
		this.candidate = (Classifier) getPreparedClassOption(this.learnerOption);
		this.candidate.resetLearning();

	}

	private void processForEnsemble1(ExecutorService executorServiceForEnsembleUpdationAndWeightUpdation) {
		executorServiceForEnsembleUpdationAndWeightUpdation.execute(new Runnable() {
			@Override
			public void run() {
				double mse_r = computeMseR();
				double candidateClassifierWeight = 1.0 / Math.exp(mse_r + Double.MIN_VALUE);
				for (int i = 0; i < learners1.length; i++) {
					double currentChunkMSI = computeMse(learners1[(int) weights1[i][1]], currentChunk);// Passive
																										// ensemble
					weights1[i][0] = 1 / Math.exp((mse_r + currentChunkMSI + (betaOption.getValue() * msi[i][0])));
				}
				addMembersToEnsemble1(candidateClassifierWeight, mse_r);

			}
		});
	}

	private ExecutorService processForEnsemble2(ExecutorService executorServiceForEnsembleUpdationAndWeightUpdation) {
		executorServiceForEnsembleUpdationAndWeightUpdation.execute(new Runnable() {
			@Override
			public void run() {
				double mse_r_warning = computeMseRForWarning();
				double candidateClassifierWeightForWarning = 1.0 / Math.exp(mse_r_warning);
				updateWeightsForWarniningEnsemble2(mse_r_warning);

				addMembersToEnsemble2(candidateClassifierWeightForWarning, mse_r_warning);
			}
		});
		return executorServiceForEnsembleUpdationAndWeightUpdation;
	}

	private void addMembersToEnsemble2(double candidateClassifierWeight, double mse_r_warning) {
		Classifier addedClassifier;
		if (this.learners2.length < this.memberCountOption.getValue()) {
			// Train and add classifier
			addedClassifier = this.addToStored2(this.candidate, candidateClassifierWeight, mse_r_warning);
		} else {
			// Substitute poorest classifier
			int poorestClassifier = this.getPoorestClassifierIndex2();

			if (this.weights2[poorestClassifier][0] < candidateClassifierWeight) {
				this.weights2[poorestClassifier][0] = candidateClassifierWeight;
				addedClassifier = this.candidate.copy();
				this.msi2[poorestClassifier][0] = mse_r_warning;
				this.learners2[(int) this.weights2[poorestClassifier][1]] = addedClassifier;
			}
		}
	}

	private void addMembersToEnsemble1(double candidateClassifierWeight, double mse_r) {
		Classifier addedClassifier;
		if (this.learners1.length < this.memberCountOption.getValue()) {
			// Train and add classifier
			addedClassifier = this.addToStored(this.candidate, candidateClassifierWeight, mse_r);
		} else {
			// Substitute poorest classifier
			int poorestClassifier = this.getPoorestClassifierIndex();

			if (this.weights1[poorestClassifier][0] < candidateClassifierWeight) {
				this.weights1[poorestClassifier][0] = candidateClassifierWeight;
				addedClassifier = this.candidate.copy();
				this.msi[poorestClassifier][0] = mse_r;//
				this.learners1[(int) this.weights1[poorestClassifier][1]] = addedClassifier;
			}
		}
	}

	private void updateWeightsForWarniningEnsemble2(double mse_r) {
		if (isDriftDetected) {
			for (int i = 0; i < learners2.length; i++) {
				double currentChunkMSI = computeMse(learners2[(int) weights2[i][1]], warningChunk);
				weights2[i][0] = 1 / Math.exp((mse_r + currentChunkMSI + (betaOption.getValue() * msi2[i][0])));
				msi2[i][0] = currentChunkMSI;
			}
		} else {
			for (int i = 0; i < learners2.length; i++) {
				double currentChunkMSI = computeMse(learners2[(int) weights2[i][1]], currentChunk);
				weights2[i][0] = 1 / Math.exp((mse_r + currentChunkMSI + (betaOption.getValue() * msi2[i][0])));
			}
		}
	}

	/**
	 * Checks if the memory limit is exceeded and if so prunes the classifiers
	 * in the ensemble.
	 */
	protected void enforceMemoryLimit(boolean isDrift) {
		double memoryLimit = this.maxByteSizeOption.getValue() / (double) (this.learners1.length + 1);

		double memoryLimit2 = this.maxByteSizeOption.getValue() / (double) (this.learners2.length + 1);
		ExecutorService executorServiceForEnforcingMemoryLimit = Executors.newFixedThreadPool(2);
		if (!isDrift) {
			executorServiceForEnforcingMemoryLimit.execute(new Runnable() {

				@Override
				public void run() {
					enforceMemoryLimitForEnsemble1(memoryLimit);
				}
			});
		}
		executorServiceForEnforcingMemoryLimit.execute(new Runnable() {

			@Override
			public void run() {
				enforceMemoryLimitForEnsemble2(memoryLimit2);
			}
		});
		executorServiceForEnforcingMemoryLimit.shutdown();
		try {
			executorServiceForEnforcingMemoryLimit.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {

		}

	}

	private void enforceMemoryLimitForEnsemble1(double memoryLimit) {
		for (int i = 0; i < this.learners1.length; i++) {
			((HoeffdingTree) this.learners1[(int) this.weights1[i][1]]).maxByteSizeOption
					.setValue((int) Math.round(memoryLimit));
			((HoeffdingTree) this.learners1[(int) this.weights1[i][1]]).enforceTrackerLimit();
		}
	}

	private void enforceMemoryLimitForEnsemble2(double memoryLimit) {
		for (int i = 0; i < this.learners2.length; i++) {
			((HoeffdingTree) this.learners2[(int) this.weights2[i][1]]).maxByteSizeOption
					.setValue((int) Math.round(memoryLimit));
			((HoeffdingTree) this.learners2[(int) this.weights2[i][1]]).enforceTrackerLimit();
		}
	}

	/**
	 * Computes the MSEr threshold.
	 * 
	 * @return The MSEr threshold.
	 */
	protected double computeMseR() {
		double p_c;
		double mse_r = 0;

		for (int i = 0; i < this.classDistributions.length; i++) {
			p_c = (double) this.classDistributions[i] / (double) this.chunkSizeOption.getValue();
			mse_r += p_c * ((1 - p_c) * (1 - p_c));
		}

		return mse_r;
	}

	protected double computeMseRForWarning() {
		double p_c;
		double mse_r = 0;

		for (int i = 0; i < this.classDistributionsForWarrning.length; i++) {
			p_c = (double) this.classDistributionsForWarrning[i] / (double) this.warningChunk.size();
			mse_r += p_c * ((1 - p_c) * (1 - p_c));
		}

		return mse_r;
	}

	/**
	 * Computes the MSE of a learner for a given chunk of examples.
	 * 
	 * @param learner
	 *            classifier to compute error
	 * @param chunk
	 *            chunk of examples
	 * @return the computed error.
	 */
	protected double computeMse(Classifier learner, Instances chunk) {
		double mse_i = 0;

		double f_ci;
		double voteSum;

		for (int i = 0; i < chunk.numInstances(); i++) {
			try {
				voteSum = 0;
				for (double element : learner.getVotesForInstance(chunk.instance(i))) {
					voteSum += element;
				}

				if (voteSum > 0) {
					f_ci = learner.getVotesForInstance(chunk.instance(i))[(int) chunk.instance(i).classValue()]
							/ voteSum;
					mse_i += (1 - f_ci) * (1 - f_ci);
				} else {
					mse_i += 1;
				}
			} catch (Exception e) {
				mse_i += 1;
			}
		}

		mse_i /= this.chunkSizeOption.getValue();

		return mse_i;
	}

	/**
	 * Adds ensemble weights to the measurements.
	 */
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measurements = new Measurement[(int) this.memberCountOption.getValue()];

		for (int m = 0; m < this.memberCountOption.getValue(); m++) {
			measurements[m] = new Measurement("Member weight " + (m + 1), -1);
		}

		if (this.weights1 != null) {
			for (int i = 0; i < this.weights1.length; i++) {
				measurements[i] = new Measurement("Member weight " + (i + 1), this.weights1[i][0]);
			}
		}

		return measurements;
	}

	/**
	 * Adds a classifier to the storage.
	 * 
	 * @param newClassifier
	 *            The classifier to add.
	 * @param newClassifiersWeight
	 *            The new classifiers weight.
	 */
	protected Classifier addToStored(Classifier newClassifier, double newClassifiersWeight, double mse_r) {
		Classifier addedClassifier = null;
		Classifier[] newStored = new Classifier[this.learners1.length + 1];
		double[][] newStoredWeights = new double[newStored.length][2];
		double[][] newmsi = new double[newStored.length][2];

		for (int i = 0; i < newStored.length; i++) {
			if (i < this.learners1.length) {
				newStored[i] = this.learners1[i];
				newStoredWeights[i][0] = this.weights1[i][0];
				newStoredWeights[i][1] = this.weights1[i][1];
				newmsi[i][0] = this.msi[i][0];
				newmsi[i][1] = this.msi[i][1];
			} else {
				newStored[i] = addedClassifier = newClassifier.copy();
				newStoredWeights[i][0] = newClassifiersWeight;
				newStoredWeights[i][1] = i;
				newmsi[i][0] = mse_r;
				newmsi[i][1] = i;
			}
		}
		this.learners1 = newStored;
		this.weights1 = newStoredWeights;
		this.msi = newmsi;

		return addedClassifier;
	}

	/**
	 * Adds a classifier to the storage.
	 * 
	 * @param newClassifier
	 *            The classifier to add.
	 * @param newClassifiersWeight
	 *            The new classifiers weight.
	 */
	protected Classifier addToStored2(Classifier newClassifier, double newClassifiersWeight, double mse_r_warning) {
		Classifier addedClassifier = null;
		Classifier[] newStored = new Classifier[this.learners2.length + 1];
		double[][] newStoredWeights = new double[newStored.length][2];
		double[][] newmsi = new double[newStored.length][2];

		for (int i = 0; i < newStored.length; i++) {
			if (i < this.learners2.length) {
				newStored[i] = this.learners2[i];
				newStoredWeights[i][0] = this.weights2[i][0];
				newStoredWeights[i][1] = this.weights2[i][1];
				newmsi[i][0] = this.msi2[i][0];
				newmsi[i][1] = this.msi2[i][1];
			} else {
				newStored[i] = addedClassifier = newClassifier.copy();
				newStoredWeights[i][0] = newClassifiersWeight;
				newStoredWeights[i][1] = i;
				newmsi[i][0] = mse_r_warning;// *************************VPN
				newmsi[i][1] = i;
			}
		}
		this.learners2 = newStored;
		this.weights2 = newStoredWeights;
		this.msi2 = newmsi;

		return addedClassifier;
	}

	/**
	 * Finds the index of the classifier with the smallest weight.
	 * 
	 * @return
	 */
	private int getPoorestClassifierIndex() {
		int minIndex = 0;

		for (int i = 1; i < this.weights1.length; i++) {
			if (this.weights1[i][0] < this.weights1[minIndex][0]) {
				minIndex = i;
			}
		}

		return minIndex;
	}

	/**
	 * Finds the index of the classifier with the smallest weight.
	 * 
	 * @return
	 */
	private int getPoorestClassifierIndex2() {
		int minIndex = 0;

		for (int i = 1; i < this.weights2.length; i++) {
			if (this.weights2[i][0] < this.weights2[minIndex][0]) {
				minIndex = i;
			}
		}

		return minIndex;
	}

	/**
	 * Initiates the current chunk and class distribution variables.
	 */
	public void initVariables() {
		super.initVariables();
		if (this.currentChunk == null) {
			this.currentChunk = new Instances(this.getModelContext());
		}
		if (this.warningChunk == null) {
			this.warningChunk = new BoastedInstances(new Instances(this.getModelContext()));
		}

		if (this.classDistributions == null) {
			this.classDistributions = new long[this.getModelContext().classAttribute().numValues()];

			for (int i = 0; i < this.classDistributions.length; i++) {
				this.classDistributions[i] = 0;
			}
		}

		if (this.classDistributionsForWarrning == null) {
			this.classDistributionsForWarrning = new long[this.getModelContext().classAttribute().numValues()];

			for (int i = 0; i < this.classDistributionsForWarrning.length; i++) {
				this.classDistributionsForWarrning[i] = 0;
			}
		}
	}

	/**
	 * Trains a component classifier on the most recent chunk of data.
	 * 
	 * @param classifierToTrain
	 *            Classifier being trained.
	 */

	private void trainingWithBagging(boolean isDrift) {
		if (!isDrift) {
			trainEnsemble1();// passive trained only if drift not there only
								// when 3*chunk cycles reached only the isDrift
								// is false
		}
		trainEnsemble2();// active is trained every time
	}

	private void trainEnsemble1() {
		for (int num = 0; num < this.currentChunk.numInstances(); num++) {
			for (int i = 0; i < this.learners1.length; i++) {
				Instance weightedInst = (Instance) this.currentChunk.instance(num).copy();
				int k = MiscUtils.poisson(this.lamdaOption.getValue(), this.classifierRandom);
				if (k > 0) {

					weightedInst.setWeight(this.currentChunk.instance(num).weight() * k);

				}
				Classifier classifierToTrain = this.learners1[(int) this.weights1[i][1]];
				classifierToTrain.trainOnInstance(weightedInst);
			}
		}
	}

	private void trainEnsemble2() {
		if (this.isDriftDetected) {
			for (int num = 0; num < this.warningChunk.numInstances(); num++) {
				for (int i = 0; i < this.learners2.length; i++) {
					Instance weightedInst = (Instance) (((BoastedInstance) this.warningChunk.instance(num)).inst)
							.copy();
					int k = MiscUtils.poisson(this.lamdaWarnOption.getValue(), this.classifierRandom);
					if (k > 0) {
						weightedInst.setWeight(this.warningChunk.instance(num).weight() * k);

					}
					Classifier classifierToTrain = this.learners2[(int) this.weights2[i][1]];

					classifierToTrain.trainOnInstance(weightedInst);
				}
			}
		} else {// if drift not detected
			for (int num = 0; num < this.currentChunk.numInstances(); num++) {
				for (int i = 0; i < this.learners2.length; i++) {
					Instance weightedInst = (Instance) this.currentChunk.instance(num).copy();
					int k = MiscUtils.poisson(this.lamdaOption.getValue(), this.classifierRandom);
					if (k > 0) {
						weightedInst.setWeight(this.currentChunk.instance(num).weight() * k);

					}
					Classifier classifierToTrain = this.learners2[(int) this.weights2[i][1]];

					classifierToTrain.trainOnInstance(weightedInst);
				}
			}
		}
	}

}
