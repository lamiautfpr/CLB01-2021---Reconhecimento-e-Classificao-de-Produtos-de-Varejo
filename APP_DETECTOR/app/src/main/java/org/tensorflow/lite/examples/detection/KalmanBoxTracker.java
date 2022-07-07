package org.tensorflow.lite.examples.detection;

import org.apache.commons.math3.filter.DefaultMeasurementModel;
import org.apache.commons.math3.filter.DefaultProcessModel;
import org.apache.commons.math3.filter.KalmanFilter;


public class KalmanBoxTracker {

    private static int count = 0;

    private KalmanFilter kf;
    public String title;
    private float time_since_update = 0;
    private int id = count;
    private int hit_streak = 0;

    public KalmanBoxTracker(String in_title, float x1, float x2, float y1, float y2){
        float width = x2 - x1;
        float height = y2 - y1;
        float cx = x1 + width/2;
        float cy = y1 + height/2;
        title = in_title;
        // matriz que define as variaveis do sistema, nesse caso: centro em x, centro em y, area , aspect ratio, velocidade em x, velocidade em y, derivada da area
        double[] initialStateEstimate = new double[]{cx, cy, Math.abs(width * height), Math.abs(width / height), 0.0, 0.0, 0.0};
        double[][] covariance = new double[][]{{   1.0,    0.0,    0.0,    0.0,    0.0,    0.0,    0.0},
                {   0.0,    1.0,    0.0,    0.0,    0.0,    0.0,    0.0},
                {   0.0,    0.0,    1.0,    0.0,    0.0,    0.0,    0.0},
                {   0.0,    0.0,    0.0,    1.0,    0.0,    0.0,    0.0},
                {   0.0,    0.0,    0.0,    0.0, 1000.0,    0.0,    0.0},
                {   0.0,    0.0,    0.0,    0.0,    0.0, 1000.0,    0.0},
                {   0.0,    0.0,    0.0,    0.0,    0.0,    0.0, 1000.0}};

        // matriz multiplicada pelo estado anterior para gerar o novo estado
        double[][] stateTransition = new double[][]{{1,0,0,0,1,0,0},{0,1,0,0,0,1,0},{0,0,1,0,0,0,1},{0,0,0,1,0,0,0},{0,0,0,0,1,0,0},{0,0,0,0,0,1,0},{0,0,0,0,0,0,1}};
        // variavel de controle do estado, 0 pois não controlamos o movimento do objeto
        double[][] control = new double[][]{{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0}};
        double[][] processNoise = new double[][]{{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
       {0.0, 0.0, 0.0, 0.0, 1.0e-02, 0.0, 0.0},
       {0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-02, 0.0},
       {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-04}};
        // Matriz de medidas, representa as váriaveis observaveis que usamos para calibrar o modelo
        // centro de x, centro de y, area e aspect ratio
        double[][] measMatrix = new double[][]{{1,0,0,0,0,0,0},{0,1,0,0,0,0,0},{0,0,1,0,0,0,0},{0,0,0,1,0,0,0}};
        double[][] measNoise = new double[][]{{ 1.0,  0.0,  0.0,  0.0},
                { 0.0,  1.0,  0.0,  0.0},
                { 0.0,  0.0, 10.0,  0.0},
                { 0.0,  0.0,  0.0, 10.0}};

        kf = new KalmanFilter(new DefaultProcessModel(stateTransition, control, processNoise, initialStateEstimate, covariance),
                new DefaultMeasurementModel(measMatrix, measNoise));

        count++;
    }

    public void update(float x1, float x2, float y1, float y2){
        time_since_update = 0;
        hit_streak++;
        float width = x2 - x1;
        float height = y2 - y1;
        float cx = x1 + width/2;
        float cy = y1 + height/2;

        double[] z = new double[]{cx, cy, Math.abs(width * height), Math.abs(width / height)};
        kf.correct(z);

    }

    public float[] predict(){
        kf.predict();
        if (time_since_update > 0) {
            hit_streak = 0;
        }
        time_since_update++;
        return get_state();
    }

    public float[] get_state(){
        double[] z = kf.getStateEstimation();
        double width = Math.sqrt(z[2]*z[3]);
        double height = z[2]/(width);
        float x1 = (float)(z[0] - width/2);
        float x2 = (float)(z[0] + width/2);
        float y1 = (float)(z[1] - height/2);
        float y2 = (float)(z[1] + height/2);
        return new float[]{x1, x2, y1, y2};
    }

    public float getTime_since_update() {
        return time_since_update;
    }

    public int getHit_streak() {
        return hit_streak;
    }

    public int getId() {
        return id;
    }
}
