package javaML_A3;

import java.io.IOException;
import java.util.Random;

public class Initialize {
	
	double[] inputs = new double[784];
	
	double[] hiddenNeurons = new double[36];
	double[] hiddenNeuronsSF = new double[hiddenNeurons.length];
	
	double[] outputs = new double[10];
	double[] outputsSF = new double[outputs.length];
	double[] desiredOuts = new double[outputs.length];
	
	//WEIGHTS
	double[][] inpTOhid = new double[hiddenNeurons.length][inputs.length];
	double[][] inpTOhidGRAD = new double[hiddenNeurons.length][inputs.length];
	
	double[][] hidTOout = new double[outputs.length][hiddenNeurons.length];
	double[][] hidTOoutGRAD = new double[outputs.length][hiddenNeurons.length];
	
	//BIAS
	double[] biasHid = new double[hiddenNeurons.length];
	double[] biasHidGRAD = new double[hiddenNeurons.length];
	
	double[] biasOut = new double[outputs.length];
	double[] biasOutGRAD = new double[outputs.length];
	
	//VALUES
	double cost = 0.0;
	double step = -0.01;
	double batchSize = 1.0;
	double totalBatches = 2000000.0;
	
	int maxOne = 0;
	double accuracy = 0;
	
	public Initialize() {
		
		Random r = new Random();
		
		for(int i = 0; i < inpTOhid.length; i++) {
			
			for(int j = 0; j < inpTOhid[0].length; j++) {
				
				inpTOhid[i][j] = r.nextDouble();
				
			}
			
		}
		
		for(int i = 0; i < hidTOout.length; i++) {
			
			for(int j = 0; j < hidTOout[0].length; j++) {
				
				hidTOout[i][j] = r.nextDouble();
				
			}
			
		}
		
		start();
		
	}

	public void start() {
		
		double[][] trainingData = null;
		int[] trainingDataLabel = null;
		
		MnistReader mr = new MnistReader();
		Random r = new Random();
		
		try {
			trainingData = mr.readData();
			trainingDataLabel = mr.readDataLabel();
			
		} catch (IOException e) {
			
			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);
		
		}
		
		for(int i = 0; i < totalBatches; i++) {
			
			for(int j = 0; j < batchSize; j++) {
				
				int item = r.nextInt(trainingData.length);
				
				desiredOuts = new double[outputs.length];
				desiredOuts[trainingDataLabel[item]] = 1.0;
				inputs = trainingData[item];
				
				forward();
				BackProp();
			}
			
		//	System.out.println(cost);
		//	System.out.println("Accuracy: " + (double) accuracy / batchSize);
			descent();
			
			hidTOoutGRAD = new double[outputs.length][hiddenNeurons.length];
			inpTOhidGRAD = new double[hiddenNeurons.length][inputs.length];
			
			biasHidGRAD = new double[hiddenNeurons.length];
			biasOutGRAD = new double[outputs.length];
			
		}
		
		System.out.println("TOTAL Accuracy: " + (double) accuracy / totalBatches);
		testSeries();
		
	}
	
	public void forward() {
		
		for(int i = 0; i < hiddenNeurons.length; i++) {
			
			hiddenNeuronsSF[i] = dot(inputs, inpTOhid[i]) + biasHid[i];
			hiddenNeurons[i] = hiddenNeuronsSF[i];//sigmoid(hiddenNeuronsSF[i]);
			
		}
		
		for(int i = 0; i < outputs.length; i++) {
			
			outputsSF[i] = dot(hiddenNeurons, hidTOout[i]) + biasOut[i];
			outputs[i] = sigmoid(outputsSF[i]);
			
		}
		
		//Calculate Cost
		cost = 0.0;
		maxOne = 0;
		
		for(int i = 0; i < outputs.length; i++) {
			
			cost += (outputs[i] - desiredOuts[i]) * (outputs[i] - desiredOuts[i]);
			
			if(outputs[maxOne] < outputs[i]) {
				
				maxOne = i;
				
			}
			
		}
		
		if(desiredOuts[maxOne] == 1.0) {
			
		//TODO	accuracy++;
			
		}
		
	}
	
	public void BackProp() {
		
		for(int i = 0; i < hidTOout.length; i++) {
			
		//	double basis0 = 2.0 * (outputs[i] - desiredOuts[i]) * sigmoidDerivative(outputs[i]);
			double basis0 = (outputs[i] - desiredOuts[i]);
			
			biasOutGRAD[i] = basis0;
			
			for(int j = 0; j < hidTOout[0].length; j++) {
				
				hidTOoutGRAD[i][j] += basis0 * hiddenNeurons[j];
				
		//		double basis1 = basis0 * inpTOhid[i][j] * sigmoidDerivative(hiddenNeuronsSF[j]);
				double basis1 = basis0 * hidTOout[i][j]; //* sigmoidDerivative(hiddenNeurons[j]);
				
				biasHidGRAD[j] += basis1;
				
				for(int k = 0; k < inpTOhid[0].length; k++) {
					
					inpTOhidGRAD[j][k] += basis1 * inputs[k];
					
				}
				
			}
			
		}
		
	}
	
	public void descent() {
		
		for(int i = 0; i < hidTOout.length; i++) {
			
			biasOutGRAD[i] = biasOutGRAD[i] / batchSize;
			biasOut[i] += step * biasOutGRAD[i];
			
			for(int j = 0; j < hidTOout[0].length; j++) {
				
				hidTOoutGRAD[i][j] = hidTOoutGRAD[i][j] / batchSize;
				hidTOout[i][j] += step * hidTOoutGRAD[i][j];
				
			}
			
		}
		
		for(int i = 0; i < inpTOhid.length; i++) {
			
			biasHidGRAD[i] = biasHidGRAD[i] / batchSize;
			biasHid[i] += step * biasHidGRAD[i];
			
			for(int j = 0; j < inpTOhid[0].length; j++) {
				
				inpTOhidGRAD[i][j] = inpTOhidGRAD[i][j] / batchSize;
				inpTOhid[i][j] += step * inpTOhidGRAD[i][j];
				
			}
			
		}
		
	}
	
	public void testSeries() {
		
		accuracy = 1000;
		
		double[][] trainingData = null;
		int[] trainingDataLabel = null;
		
		MnistReader mr = new MnistReader();
		Random r = new Random();
		
		try {
			trainingData = mr.readDataTest();
			trainingDataLabel = mr.readDataLabelTest();
			
		} catch (IOException e) {
			
			e.printStackTrace();
			System.err.print("Closing");
			System.exit(1);
		
		}
		
		for(int i = 0; i < 1000; i++) {
			
			int item = r.nextInt(trainingData.length);
			inputs = trainingData[item];
			forward();
			if(maxOne != trainingDataLabel[item]) {
				System.out.print("(MISTAKE)");
				accuracy -= 1;
			}
			
			System.out.println("Predicted: " + maxOne + " Real: " + trainingDataLabel[item] + " Item: " + item);
			
		}
		
		System.out.println(accuracy);
		
	}
	
	public double sigmoid(double value) {
		
		return 1.0 / (1.0 + (1.0 / Math.exp(value)));
		
	}
	
	public double sigmoidDerivative(double value) {
		
		return sigmoid(value) * (1.0 - sigmoid(value));
		
	}
	
	public double dot(double[] arr1, double[] arr2) {
		
		double value = 0.0;
		
		for(int i = 0; i < arr1.length; i++) {
			
			value += arr1[i] * arr2[i];
			
		}
		
		return value;
		
	}
	
}
