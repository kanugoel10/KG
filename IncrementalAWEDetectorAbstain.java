/*
 *    AccuracyWeightedEnsemble.java
 *    Copyright (C) 2010 Poznan University of Technology, Poznan, Poland
 *    @author Dariusz Brzezinski (dariusz.brzezinski@cs.put.poznan.pl)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import moa.classifiers.Classifier;
import moa.classifiers.core.driftdetection.DriftDetectionMethod;
import moa.core.BoastedInstance;
import moa.core.BoastedInstances;
import moa.core.DoubleVector;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import moa.tasks.TaskMonitor;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class IncrementalAWEDetectorAbstain extends AccuracyWeightedEnsemble {

	private static final long serialVersionUID = 1L;

	public FloatOption betaOption = new FloatOption("betaOption", 'z', "Lamda 2 value.", 0.8, 0.0, 1);

	public ClassOption detectorLearnerOption = new ClassOption("detectorBaseLearner", 'b', "Classifier to train.",
			Classifier.class, "bayes.NaiveBayes");

	public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'd',
			"Drift detection method to use.", DriftDetectionMethod.class, "DDM2");

	public IntOption threshHoldOption = new IntOption("threshHoldOption", 'e', "threshHoldOption", 300, 200, 450);// Alpha
																													// threshold

	public IntOption threshHoldTwoOption = new IntOption("threshHoldTwoOption", 's', "Relative Difference count", 30,
			10, 50);

	public FloatOption boostOption = new FloatOption("boostOption", 'u', "boost Option", 2.0, 0.0, 4.0);

	protected Classifier detectorClassifier;
	protected Classifier alarmDetectorClassifier;
	protected DriftDetectionMethod driftDetectionMethod;
	protected boolean newClassifierReset;
	protected boolean isAlarm;
	protected boolean isWarning = false;
	protected boolean isDriftDetected;
	protected int ddmLevel;
	protected double detectorWeight;
	protected BoastedInstances warningChunk;
	protected int correctCount1;
	protected int noOfFalseAlarmsDetected = 0; 
	protected int correctCount2;
	protected boolean useEnsemble1 = true;
	protected boolean useBothEnsemble = false;
	protected int correctCountWindow;
	protected long[] classDistributionsForWarrning;
	@Override
	public void prepareForUseImpl(TaskMonitor monitor,
			ObjectRepository repository) {
		super.prepareForUseImpl(monitor, repository);

		this.detectorClassifier = ((Classifier) getPreparedClassOption(this.detectorLearnerOption))
				.copy();
		this.detectorClassifier.resetLearning();
		this.alarmDetectorClassifier = ((Classifier) getPreparedClassOption(this.detectorLearnerOption))
				.copy();
		this.alarmDetectorClassifier.resetLearning();
		this.driftDetectionMethod = ((DriftDetectionMethod) getPreparedClassOption(this.driftDetectionMethodOption))
				.copy();
		this.newClassifierReset = false;
		this.isAlarm = false;
		this.isDriftDetected = false;
		this.detectorWeight = 1;
	}

	@Override
	public void resetLearningImpl() {
		super.resetLearningImpl();

		this.detectorClassifier = ((Classifier) getPreparedClassOption(this.detectorLearnerOption))
				.copy();
		this.detectorClassifier.resetLearning();
		this.alarmDetectorClassifier = ((Classifier) getPreparedClassOption(this.detectorLearnerOption))
				.copy();
		this.alarmDetectorClassifier.resetLearning();
		this.driftDetectionMethod = ((DriftDetectionMethod) getPreparedClassOption(this.driftDetectionMethodOption))
				.copy();
		this.newClassifierReset = false;
		this.isAlarm = false;
		this.isDriftDetected = false;
		this.detectorWeight = 1;
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
			this.initVariables();

		this.classDistributionsForWarrning[(int) inst.classValue()]++;
		this.classDistributions[(int) inst.classValue()]++;
		this.currentChunk.add(inst);
		this.detectorClassifier.trainOnInstance(inst);//training on each instance
		this.processedInstances++;
			int trueClass = (int) inst.classValue();
		boolean prediction;
		if (Utils.maxIndex(this.detectorClassifier.getVotesForInstance(inst)) == trueClass) {
			prediction = true;
		} else {
			prediction = false;
		}
		this.ddmLevel = this.driftDetectionMethod.computeNextVal(prediction);

		switch (this.ddmLevel) {
		case DriftDetectionMethod.DDM_WARNING_LEVEL:
			isWarning=true;
			if (newClassifierReset == true) {
				this.alarmDetectorClassifier.resetLearning();
				newClassifierReset = false;
			}
			BoastedInstance inst2=new BoastedInstance((Instance)inst.copy());
			//TODO: make factor
			inst2.setWeight(inst.weight()*boostOption.getValue());
			inst2.setWarningInstance(true);
		
			this.warningChunk.add(inst2);
			this.alarmDetectorClassifier.trainOnInstance(inst);
			break;

		case DriftDetectionMethod.DDM_OUTCONTROL_LEVEL://drift detected !!
			
			
			this.correctCount1=0; 
			this.correctCount2=0;
			this.isAlarm = true;
			this.isDriftDetected = true;
			this.detectorClassifier = null;
			this.detectorClassifier = this.alarmDetectorClassifier;
			if (this.detectorClassifier instanceof WEKAClassifier) {
				((WEKAClassifier) this.detectorClassifier).buildClassifier();
			}

			this.detectorWeight = this.computeWeight(this.detectorClassifier, this.currentChunk);
		
			this.processChunk(true,false);
			this.alarmDetectorClassifier = ((Classifier) getPreparedClassOption(this.detectorLearnerOption)).copy();
			this.alarmDetectorClassifier.resetLearning();

			// reset chunk
    		this.classDistributions = null;
    		this.classDistributionsForWarrning = null;
    	
    		this.warningChunk = null;
    		this.initVariables();
    		this.classDistributions[(int) inst.classValue()]++;
    		this.currentChunk.add(inst);
    		this.detectorClassifier.trainOnInstance(inst);
    	    		this.isDriftDetected = false;
			this.isAlarm = false;
			isWarning=false;
			break;

		case DriftDetectionMethod.DDM_INCONTROL_LEVEL:  //no drift
			if(isWarning){
				for(Instance instance:this.warningChunk){
					if(instance instanceof BoastedInstance && ((BoastedInstance) instance).isWarningInstance()){
						((BoastedInstance) instance).setWarningInstance(false);
						instance.setWeight(instance.weight()/2);
					}
				}
				isWarning=false;//****false alarm
				this.noOfFalseAlarmsDetected=this.noOfFalseAlarmsDetected+1;
			}
			newClassifierReset = true;
			break;
		}
		
		if (!isWarning) {
			this.warningChunk.add(new BoastedInstance((Instance)inst));
		}
	
		if (this.processedInstances % this.chunkSize == 0) { // if chunk size reached
			//TODO make 
			if(this.processedInstances % this.chunkSize * 3 == 0){
				this.processChunk(false,true);//Passive updation
			}else{
				this.processChunk(false,false);}
			determinVotingLogic();
				correctCount1=0;
			correctCount2=0;
			
		}
	}

	/**
	 * <code>determinVotingLogic</code> Called after the current chunk has been
	 * processed. Determines the voting logic for the next testing phase
	 **/
	private void determinVotingLogic() {
		
		if((correctCount1>threshHoldOption.getValue() && correctCount2>threshHoldOption.getValue()) && (Math.abs(correctCount1-correctCount2)<threshHoldTwoOption.getValue())) {
			this.useBothEnsemble=true;
	
		}

		else if(correctCount1>=correctCount2){
			this.useEnsemble1=true;
			this.useBothEnsemble=false;
		
		}else{
			this.useEnsemble1=false;
			this.useBothEnsemble=false;
		
		}
	}

	/**
	 * Initiates the current chunk and class distribution variables.
	 */
	protected void initVariables() {
		if (this.currentChunk == null) {
			this.currentChunk = new Instances(this.getModelContext());
		}
		if (this.warningChunk == null) {
			this.warningChunk = new BoastedInstances(new Instances(this.getModelContext()));
		}
		if (this.classDistributions == null) {
			this.classDistributions = new long[this.getModelContext()
					.classAttribute().numValues()];

			for (int i = 0; i < this.classDistributions.length; i++) {
				this.classDistributions[i] = 0;
			}
		}
		
		if (this.classDistributionsForWarrning == null) {
			this.classDistributionsForWarrning = new long[this.getModelContext()
					.classAttribute().numValues()];

			for (int i = 0; i < this.classDistributionsForWarrning.length; i++) {
				this.classDistributionsForWarrning[i] = 0;
			}
		}
	}
	

	protected void processChunk(boolean isDrift,boolean isReplace) {
        // Compute weights
		double candidateClassifierWeight=0;
		if (isDriftDetected) {
			processWhenDrift(candidateClassifierWeight);

		} else {
			processWhenNoDrift(candidateClassifierWeight);

		}


        int ensembleSize = java.lang.Math.min(this.storedLearners.length, this.maxMemberCount);
        this.ensemble = new Classifier[ensembleSize];
        this.ensembleWeights = new double[ensembleSize];

        java.util.Arrays.sort(this.storedWeights, weightComparator);

        // Select top k classifiers to construct the ensemble
        int storeSize = this.storedLearners.length;
        for (int i = 0; i < ensembleSize; i++) {
            this.ensembleWeights[i] = this.storedWeights[storeSize - i - 1][0];
            this.ensemble[i] = this.storedLearners[(int) this.storedWeights[storeSize - i - 1][1]];
        }

        this.classDistributions = null;
        this.currentChunk = null;
        this.candidateClassifier = (Classifier) getPreparedClassOption(this.learnerOption);
        this.candidateClassifier.resetLearning();
    }

	private void processWhenNoDrift(double candidateClassifierWeight) {
		if (this.currentChunk.numInstances() > this.numFolds) {
			candidateClassifierWeight = this.computeCandidateWeight(this.candidateClassifier, this.currentChunk,
					this.numFolds);
		} else {
			candidateClassifierWeight = 1;
		}
		for (int i = 0; i < this.storedLearners.length; i++) {
			this.storedWeights[i][0] = this.computeWeight(this.storedLearners[(int) this.storedWeights[i][1]],
					this.currentChunk);
		}
		if (this.storedLearners.length < this.maxStoredCount) {
			// Train and add classifier
			for (int num = 0; num < this.currentChunk.numInstances(); num++) {
				this.candidateClassifier.trainOnInstance(this.currentChunk.instance(num));
			}

			this.addToStored(this.candidateClassifier, candidateClassifierWeight);
		} else {
			// Substitute poorest classifier
			java.util.Arrays.sort(this.storedWeights, weightComparator);

			if (this.storedWeights[0][0] < candidateClassifierWeight) {
				for (int num = 0; num < this.currentChunk.numInstances(); num++) {
					this.candidateClassifier.trainOnInstance(this.currentChunk.instance(num));
				}
				this.storedWeights[0][0] = candidateClassifierWeight;
				this.storedLearners[(int) this.storedWeights[0][1]] = this.candidateClassifier.copy();
			}
		}
	}

	private void processWhenDrift(double candidateClassifierWeight) {
		if (this.warningChunk.numInstances() > this.numFolds) {//!!!!!!!!!!!! warningChunk pe opeartion
			candidateClassifierWeight = this.computeCandidateWeight(this.candidateClassifier, this.warningChunk,
					this.numFolds);
		} else {
			candidateClassifierWeight = 1;
		}
		for (int i = 0; i < this.storedLearners.length; i++) {
			this.storedWeights[i][0] = this.computeWeight(this.storedLearners[(int) this.storedWeights[i][1]],
					this.warningChunk);
		}
		if (this.storedLearners.length < this.maxStoredCount) {
			// Train and add classifier
			for (int num = 0; num < this.warningChunk.numInstances(); num++) {
				this.candidateClassifier.trainOnInstance(this.warningChunk.instance(num));
			}

			this.addToStored(this.candidateClassifier, candidateClassifierWeight);
		} else {
			// Substitute poorest classifier
			java.util.Arrays.sort(this.storedWeights, weightComparator);

			if (this.storedWeights[0][0] < candidateClassifierWeight) {
				for (int num = 0; num < this.warningChunk.numInstances(); num++) {
					this.candidateClassifier.trainOnInstance(this.warningChunk.instance(num));
				}
				this.storedWeights[0][0] = candidateClassifierWeight;
				this.storedLearners[(int) this.storedWeights[0][1]] = this.candidateClassifier.copy();
			}
		}
	}

	/**
	 * Computes the weight of a given classifier.
	 * 
	 * @param learner
	 *            Classifier to calculate weight for.
	 * @param chunk
	 *            Data chunk of examples.
	 * @param useMseR
	 *            Determines whether to use the MSEr threshold.
	 * @return The given classifier's weight.
	 */
	protected double computeWeight(Classifier learner, Instances chunk) {
		double mse_i = 0;
		double mse_r = 0;

		double f_ci;
		double voteSum;

		for (int i = 0; i < chunk.numInstances(); i++) {
			try {
				voteSum = 0;
				for (double element : learner.getVotesForInstance(chunk
						.instance(i))) {
					voteSum += element;
				}

				if (voteSum > 0) {
					f_ci = learner.getVotesForInstance(chunk.instance(i))[(int) chunk
							.instance(i).classValue()] / voteSum;
					mse_i += (1 - f_ci) * (1 - f_ci);
				} else {
					mse_i += 1;
				}
			} catch (Exception e) {
				mse_i += 1;
			}
		}

		mse_i /= chunk.numInstances();
		mse_r = this.computeMseR();

		return java.lang.Math.max(mse_r - mse_i, 0);
	}
	
	/**
	 * Predicts a class for an example.
	 */
	@Override
	public double[] getVotesForInstance(Instance inst) {
		DoubleVector combinedVote = new DoubleVector();

		if (this.trainingWeightSeenByModel > 0.0) {
			for (int i = 0; i < this.ensemble.length; i++) {
				if (this.ensembleWeights[i] > 0.0) {
					DoubleVector vote = new DoubleVector(
							this.ensemble[i].getVotesForInstance(inst));

					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						// scale weight and prevent overflow
						vote.scaleValues(this.ensembleWeights[i]
								/ (1.0 * this.ensemble.length + 2));
						combinedVote.addValues(vote);
					}
				}
			}

			DoubleVector detectorVote = new DoubleVector(
					this.detectorClassifier.getVotesForInstance(inst));

			if (detectorVote.sumOfValues() > 0.0) {
				detectorVote.normalize();
			
				detectorVote.scaleValues(this.detectorWeight
						/ (1.0 * this.ensemble.length + 2));
				combinedVote.addValues(detectorVote);
			}
		}
		combinedVote.normalize();
		return combinedVote.getArrayRef();
	}

	public int getNoOfFalseAlarmsDetected() {
		return noOfFalseAlarmsDetected;
	}
	
	
	
}
