/**
 * Created by Solomon on 7/5/2017.
 */
public class LinearRegression {

    double dHypothesis[];

    // Load Population data into Array X
    double[][] dInputData = new double[][]{{1,6.1101},{1,5.5277},{1,8.5186},{1,7.0032},{1,5.8598},{1,8.3829},{1,7.4764},{1,8.5781},{1,6.4862},{1,5.0546},{1,5.7107},{1,14.164},{1,5.734},{1,8.4084},{1,5.6407},{1,5.3794},{1,6.3654},{1,5.1301},{1,6.4296},{1,7.0708},{1,6.1891},{1,20.27},{1,5.4901},{1,6.3261},{1,5.5649},{1,18.945},{1,12.828},{1,10.957},{1,13.176},{1,22.203},{1,5.2524},{1,6.5894},{1,9.2482},{1,5.8918},{1,8.2111},{1,7.9334},{1,8.0959},{1,5.6063},{1,12.836},{1,6.3534},{1,5.4069},{1,6.8825},{1,11.708},{1,5.7737},{1,7.8247},{1,7.0931},{1,5.0702},{1,5.8014},{1,11.7},{1,5.5416},{1,7.5402},{1,5.3077},{1,7.4239},{1,7.6031},{1,6.3328},{1,6.3589},{1,6.2742},{1,5.6397},{1,9.3102},{1,9.4536},{1,8.8254},{1,5.1793},{1,21.279},{1,14.908},{1,18.959},{1,7.2182},{1,8.2951},{1,10.236},{1,5.4994},{1,20.341},{1,10.136},{1,7.3345},{1,6.0062},{1,7.2259},{1,5.0269},{1,6.5479},{1,7.5386},{1,5.0365},{1,10.274},{1,5.1077},{1,5.7292},{1,5.1884},{1,6.3557},{1,9.7687},{1,6.5159},{1,8.5172},{1,9.1802},{1,6.002},{1,5.5204},{1,5.0594},{1,5.7077},{1,7.6366},{1,5.8707},{1,5.3054},{1,8.2934},{1,13.394},{1,5.4369}};

    // Load profit data into y
    double[] dOutputData = {17.592, 9.1302, 13.662, 11.854, 6.8233, 11.886, 4.3483, 12, 6.5987, 3.8166, 3.2522, 15.505, 3.1551, 7.2258, 0.71618, 3.5129, 5.3048, 0.56077, 3.6518, 5.3893, 3.1386, 21.767, 4.263, 5.1875, 3.0825, 22.638, 13.501, 7.0467, 14.692, 24.147, -1.22, 5.9966, 12.134, 1.8495, 6.5426, 4.5623, 4.1164, 3.3928, 10.117, 5.4974, 0.55657, 3.9115, 5.3854, 2.4406, 6.7318, 1.0463, 5.1337, 1.844, 8.0043, 1.0179, 6.7504, 1.8396, 4.2885, 4.9981, 1.4233, -1.4211, 2.4756, 4.6042, 3.9624, 5.4141, 5.1694, -0.74279, 17.929, 12.054, 17.054, 4.8852, 5.7442, 7.7754, 1.0173, 20.992, 6.6799, 4.0259, 1.2784, 3.3411, -2.6807, 0.29678, 3.8845, 5.7014, 6.7526, 2.0576, 0.47953, 0.20421, 0.67861, 7.5435, 5.3436, 4.2415, 6.7981, 0.92695, 0.152, 2.8214, 1.8451, 4.2959, 7.2029, 1.9869, 0.14454, 9.0551, 0.61705};

    // Initialize theta array
    double[] dTheta = {0.0, 0.0};

    // Number of Iterations
    int iIterations = 1500;

    // Learning rate
    double dLearningRate = 0.01;

    // number of training examples
    int iNumberOfTrainingExamples = dOutputData.length;

    public static void main(String args[]){

        LinearRegression objLinearRegression = new LinearRegression();

        objLinearRegression.calculateHypothesis();

        objLinearRegression.gradientDescent();

        objLinearRegression.computeSalesBasedOnPopulation();
    }

    void calculateHypothesis(){

        // dHypothesis = dInputData * dTheta
        dHypothesis = new double[iNumberOfTrainingExamples];

        // Initialize dHypothesis to zero vector
        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            dHypothesis[index1] = 0.0;
        }


        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            for(int index2=0; index2<dInputData[0].length; index2++){
                dHypothesis[index1] = dHypothesis[index1] + (dInputData[index1][index2] * dTheta[index2]);
            }
        }

        //computeCost();
    }

    void computeCost(){

        double dSumOfSquareErrors = 0.0;
        double dDeviation = 0.0;

        for(int index1=0; index1<iNumberOfTrainingExamples; index1++){
            dDeviation = dHypothesis[index1] - dOutputData[index1];
            dSumOfSquareErrors = dSumOfSquareErrors + (dDeviation * dDeviation);
        }

        //System.out.println(dSumOfSquareErrors);

        double dComputedCost =  (1.0 / (2.0 * iNumberOfTrainingExamples) ) * dSumOfSquareErrors;

        //System.out.println(dComputedCost);

    }

    void gradientDescent(){

        double dDeviation[] = new double[iNumberOfTrainingExamples];

        double dTransposedInputData[][] = transposeMatrix(dInputData);
        double dDelta[] = new double[dTransposedInputData.length];

        for(int index2=0; index2<dTransposedInputData.length; index2++){
            dDelta[index2]= 0.0;
        }

        for(int index1=0; index1<iIterations; index1++){
            calculateHypothesis();

            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dDeviation[index2] = dHypothesis[index2] - dOutputData[index2];
            }

            for(int index2=0; index2<dTransposedInputData.length; index2++){
                for(int index3=0; index3<dTransposedInputData[0].length; index3++){
                    dDelta[index2] = dDelta[index2] + (dTransposedInputData[index2][index3] * dDeviation[index3]);
                }
                dDelta[index2] = dDelta[index2] / iNumberOfTrainingExamples;
                dDelta[index2] = dLearningRate * dDelta[index2];
            }

            for(int index2=0; index2<dTheta.length; index2++){
                dTheta[index2] = dTheta[index2] - dDelta[index2];
            }

            computeCost();

        }

        for(int index1=0; index1<dTheta.length; index1++){
            System.out.println(dTheta[index1]);
        }
    }

    double[][] transposeMatrix(double dInputMatrix[][]){

        double dTransposedMatrix[][] = new double[dInputMatrix[0].length][dInputMatrix.length];

        for(int index1=0; index1<dInputData[0].length; index1++){
            for(int index2=0; index2<iNumberOfTrainingExamples; index2++){
                dTransposedMatrix[index1][index2] = dInputMatrix[index2][index1];
            }
        }

        return dTransposedMatrix;

    }

    void computeSalesBasedOnPopulation(){
        double[][] dPopulationInputData = {{1.0, 7.0}};
        double dPredictedProfit = 0.0;

        for(int index1=0; index1<dPopulationInputData.length; index1++){
            for(int index2=0; index2<dPopulationInputData[0].length; index2++){
                dPredictedProfit = dPredictedProfit + (dPopulationInputData[index1][index2] * dTheta[index2]);
            }
        }

        System.out.println("For population of 70000, profit predicted is :" + dPredictedProfit * 10000);


    }

}
