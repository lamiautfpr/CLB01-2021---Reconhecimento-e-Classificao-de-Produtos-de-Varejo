package org.tensorflow.lite.examples.detection;

import android.util.Pair;

import java.util.ArrayList;
import java.util.List;

public class HungarianAlgorithm {
    private int[][] costMatrix;
    private boolean[][] covered;
    private int originalRowLength;
    private int originalColumnLength;
    public HungarianAlgorithm(int[][] cost){
        originalRowLength = cost.length;
        originalColumnLength = cost[0].length;
        if(originalRowLength == originalColumnLength)
            costMatrix = cost;
        else{
            int len = originalRowLength > originalColumnLength ? originalRowLength : originalColumnLength;
            costMatrix = new int[len][len];
            for (int i = 0; i < originalRowLength; i++) {
                for (int j = 0; j < originalColumnLength; j++) {
                    costMatrix[i][j] = cost[i][j];
                }
            }
        }
        covered = new boolean[2][costMatrix.length];
        int min = costMatrix[0][0];
        for (int[] matrix : costMatrix) {
            for (int j = 0; j < costMatrix.length; j++) {
                if (matrix[j] < min) {
                    min = matrix[j];
                }
            }
        }
        if (min >= 0){
            return;
        }
        for (int i = 0; i < costMatrix.length; i++) {
            for (int j = 0; j < costMatrix.length; j++) {
                costMatrix[i][j] -= min;
            }
        }
    }
    public int[] assign(){
        subtractRowMinima();
        subtractColumnMinima();
        do{
            coverZeros();
            createZeros();
        }while(sumCoveredLines() < costMatrix.length);
        int[] assignment = optimalAssignment();
        int[] results = new int[originalRowLength];
        for (int i = 0; i < results.length; i++) {
            if (assignment[i] < originalColumnLength)
                results[i] = assignment[i];
            else
                results[i] = -1;
        }
        return results;
    }
    private void subtractRowMinima(){
        int min;
        for (int i = 0; i < costMatrix.length; i++) {
            min = costMatrix[i][0];
            for (int j = 1; j < costMatrix.length; j++) {
                if(costMatrix[i][j] < min){
                    min = costMatrix[i][j];
                }
            }
            for (int j = 0; j < costMatrix.length; j++) {
                costMatrix[i][j] -= min;
            }
        }

    }
    private void subtractColumnMinima(){
        int min;
        for (int i = 0; i < costMatrix.length; i++) {
            min = costMatrix[0][i];
            for (int j = 1; j < costMatrix.length; j++) {
                if(costMatrix[j][i] < min){
                    min = costMatrix[j][i];
                }
            }
            for (int j = 0; j < costMatrix.length; j++) {
                costMatrix[j][i] -= min;
            }
        }
    }
    private void coverZeros(){
        for (int i = 0; i < costMatrix.length; i++) {
            for (int j = 0; j < costMatrix.length; j++) {
                if (costMatrix[i][j] == 0 && !covered[1][j]){
                    costMatrix[i][j] = -1;
                    covered[1][j] = true;
                    break;
                }
            }
        }
        List<Pair<Integer, Integer>> path = new ArrayList<>();
        for (int i = 0; i < costMatrix.length; i++) {
            for (int j = 0; j < costMatrix.length; j++) {
                if (costMatrix[i][j] == 0 && !covered[1][j]){
                    costMatrix[i][j] = -2;
                    int col = starredZeroInRow(i);
                    if (col >= 0){
                        covered[0][i] = true;
                        covered[1][col] = false;
                        break;
                    } else{
                        int row = starredZeroInColumn(col);
                        col = j;
                        while(row > 0){
                            path.add(new Pair<>(row, col));
                            col = primedZeroInRow(row);
                            path.add(new Pair<>(row, col));
                            row = starredZeroInColumn(col);
                        }
                        for (Pair<Integer, Integer> elem : path) {
                            if (costMatrix[elem.first][elem.second] == -1) {
                                costMatrix[elem.first][elem.second] = -2;
                            } else if (costMatrix[elem.first][elem.second] == -2) {
                                costMatrix[elem.first][elem.second] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    private void createZeros(){
        int min = 1000;
        for (int i = 0; i < costMatrix.length; i++) {
            if(covered[0][i]){
                continue;
            }
            for (int j = 0; j < costMatrix.length; j++) {
                if(covered[1][j]){
                    continue;
                }
                if(costMatrix[i][j] < min){
                    min = costMatrix[i][j];
                }
            }
        }
        for (int i = 0; i < costMatrix.length; i++) {
            for (int j = 0; j < costMatrix.length; j++) {
                if(covered[1][j] && covered[0][i]){
                    costMatrix[i][j] += min;
                } else if(!covered[0][i] && !covered[1][j]) {
                    costMatrix[i][j] -= min;
                }
            }
        }
    }
    private int starredZeroInRow(int row){
        for (int i = 0; i < costMatrix.length; i++) {
            if (costMatrix[row][i] == -1){
                return i;
            }
        }
        return -1;
    }
    private int starredZeroInColumn(int column){
        for (int i = 0; i < costMatrix.length; i++) {
            if (costMatrix[i][column] == -1){
                return i;
            }
        }
        return -1;
    }
    private int primedZeroInRow(int row){
        for (int i = 0; i < costMatrix.length; i++) {
            if (costMatrix[row][i] == -2){
                return i;
            }
        }
        return -1;
    }
    private int sumCoveredLines(){
        int sum = 0;
        for (boolean[] row : covered) {
            for (boolean col : row) {
                if (col) {
                    sum++;
                }
            }
        }
        return sum;
    }
    private int[] optimalAssignment(){
        int[] assignments = new int[costMatrix.length];
        for (int i = 0; i < costMatrix.length; i++) {
            for (int j = 0; covered[0][i] && j < costMatrix.length; j++) {
                if (costMatrix[i][j] <= 0){
                    costMatrix[i][j] = 1000;
                    assignments[i] = j;
                    covered[0][i] = false;
                    if (!covered[1][j]){
                        for (int k = 0; k < costMatrix.length; k++) {
                            if (costMatrix[k][j] <= 0){
                                costMatrix[k][j] = 1000;
                            }
                        }
                    }
                }

            }
            for (int j = 0; covered[1][i] && j < costMatrix.length; j++) {
                if (costMatrix[j][i] <= 0){
                    costMatrix[j][i] = 1000;
                    assignments[j] = i;
                    covered[1][i] = false;
                    if (!covered[0][j]){
                        for (int k = 0; k < costMatrix.length; k++) {
                            if (costMatrix[j][k] <= 0){
                                costMatrix[j][k] = 1000;
                            }
                        }
                    }
                }
            }
        }
        return assignments;
    }
}
