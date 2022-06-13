package org.tensorflow.lite.examples.detection;


import android.graphics.RectF;
import android.util.Pair;

import org.tensorflow.lite.examples.detection.tflite.Detector;

import java.util.ArrayList;
import java.util.List;



/**
 * Essa classe implementa o método de object tracking Simple Online RealTime Tracking, encontrado em https://github.com/abewley/sort
 */

public class SORT {

    private int max_age;
    private int min_hits;
    private float iou_threshold;
    private List<KalmanBoxTracker> trackers = new ArrayList<KalmanBoxTracker>();
    private int frame_count = 0;
    public SORT(int max_age_in, int min_hits_in, float iou_threshold_in){
        max_age = max_age_in;
        min_hits = min_hits_in;
        iou_threshold = iou_threshold_in;
    }

    public  List<Pair<Integer, float[]>> update(List<Detector.Recognition> detections){
        frame_count++;
        List<Pair<Integer, float[]>> rets = new ArrayList<Pair<Integer, float[]>>();
        if (detections.isEmpty()){
            return rets;
        }else if(trackers.isEmpty()){
            for (Detector.Recognition det : detections) {
                RectF l = det.getLocation();
                // Por algum motivo o top representa o menor valor y da bounding box
                KalmanBoxTracker k = new KalmanBoxTracker(det.getTitle(), l.left, l.right, l.top, l.bottom);
                trackers.add(k);
                rets.add(new Pair<>(k.getId(), new float[]{l.left, l.right, l.top, l.bottom}));
            }
            return rets;
        }

        float[][] pred_locations = new float[trackers.size()][4];
        float[] pos = new float[]{0, 0, 0, 0};
        for (int i=0; i < trackers.size(); i++){
            pos = trackers.get(i).predict();
            pred_locations[i] = new float[]{pos[0], pos[1], pos[2], pos[3]};
        }
        float[][] locations = new float[detections.size()][4];
        for (int i = 0; i < detections.size(); i++) {
            RectF l = detections.get(i).getLocation();
            locations[i][0] = l.left;
            locations[i][1] = l.right;
            locations[i][2] = l.top;
            locations[i][3] = l.bottom;
        }
        int[] results = associate_detections_to_trackers(locations, pred_locations);
        if (results == null){
            return rets;
        }
        float[] l;
        for (int i=0; i < results.length; i++){
            l = locations[i];
            if (results[i] != -1) {
                trackers.get(results[i]).update(l[0], l[1], l[2], l[3]);
            } else{
                // se não possuir um tracker correspondente, instancia um novo
                trackers.add(new KalmanBoxTracker(detections.get(i).getTitle(), l[0], l[1], l[2], l[3]));
            }
        }

        KalmanBoxTracker t;
        for (int i = trackers.size()-1; i >= 0; i--){
            t = trackers.get(i);
            if (t.getTime_since_update() < 10){
                rets.add(new Pair<>(t.getId(), t.get_state()));
            }
            if (t.getTime_since_update() > max_age){
                trackers.remove(i);
            }
        }
        return rets;
    }

    public int[] associate_detections_to_trackers(float[][] detections, float[][] predictions){
/*
    Calcula inteseccao sobre uniao entre as deteccoes e as previsões para em seguida associar as
    deteccoes e previsoes com maior IOU atraves do hungarian algorithm
 */
    if(detections.length == 0 || predictions.length == 0){
        return null;
    }
    int[][] iou_matrix = new int[detections.length][predictions.length];
    for(int i=0;i < detections.length; i++){
        for (int j = 0; j < predictions.length; j++) {
            iou_matrix[i][j] = -Math.round(calculateIOU(detections[i], predictions[j])*100);
        }
    }
    int[] assignment = new HungarianAlgorithm(iou_matrix).assign();

    return assignment;
    }
    public float calculateIOU(float[] boxA, float[] boxB) {
          /*
             Calcula interseccao sobre união de duas bounding boxes na forma [x1, x2, y1, y2]
         */
        float x1A = Math.min(boxA[0], boxA[1]);
        float x2A = Math.max(boxA[0], boxA[1]);
        float y1A = Math.min(boxA[2], boxA[3]);
        float y2A = Math.max(boxA[2], boxA[3]);

        float x1B = Math.min(boxB[0], boxB[1]);
        float x2B = Math.max(boxB[0], boxB[1]);
        float y1B = Math.min(boxB[2], boxB[3]);
        float y2B = Math.max(boxB[2], boxB[3]);

        float x1 = Math.max(x1A, x1B);
        float x2 = Math.min(x2A, x2B);
        float y1 = Math.max(y1A, y1B);
        float y2 = Math.min(y2A, y2B);

        float interArea = Math.max(0 , x2 - x1 + 1) * Math.max(0 , y2 - y1 + 1);
        if(interArea == 0){
            return 0;
        }
        float boxAArea = (x2A - x1A + 1) * (y2A - y1A + 1);
        float boxBArea = (x2B - x1B + 1) * (y2B - y1B + 1);

        float iou = interArea / (boxAArea + boxBArea - interArea);
        return iou;
    }
}
