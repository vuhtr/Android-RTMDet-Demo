package com.example.rtmdet_ins_snapedit;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.Build;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;
import kotlin.io.ByteStreamsKt;

public class ObjectDetector {
    // constant of current model family
    private static final int PAD_VAL = 114;
    private static final float BOX_IOU_THRES = 0.7F;
    private static final float MASK_IOU_THRES = 0.7F;
    private static final float OVERLAP_THRES = 0.8F;
    private static final float EPS = 1e-6F;

    private static final float[] MEAN = {103.53F, 116.28F, 123.675F};
    private static final float[] STD = {57.375F, 57.12F, 58.395F};

    // constant of post-processing
    private static final int BOX_THRES = 20;    // ignore too small boxes


    static class DetectionResult {
        public ArrayList<int[]> boxes;      // (n, 4) - format x1, y1, x2, y2
        public ArrayList<Bitmap> masks;    // (n,) - bitmap of mask corresponding to box (size of mask = size of box)
        public ArrayList<Float> scores;     // (n, ) - confidence score between 0 and 1
        public ArrayList<String> labels;    // (n, ) - class label

        public DetectionResult(ArrayList<int[]> boxes, ArrayList<Bitmap> masks, ArrayList<Float> scores, ArrayList<String> labels) {
            this.boxes = boxes;
            this.masks = masks;
            this.scores = scores;
            this.labels = labels;
        }
    }


    private Resources resources;
    private HashMap<Integer, String> classMapping;
    private OrtEnvironment ortEnv;  // ONNX runtime environment
    private OrtSession ortSession;  // ONNX runtime session
    private final int inferSize;      // input size of the model
    private final float commonThres;  // confidence threshold for common bounding box
    private final float personThres;  // confidence threshold for person (special case)

    public ObjectDetector(Resources resources, int classesFileID, int modelID, int inferSize, float commonThres, float personThres) {
        this.resources = resources;
        this.inferSize = inferSize;
        this.commonThres = commonThres;
        this.personThres = personThres;
        readClasses(classesFileID);
        createOrtSession(modelID);
    }

    private void createOrtSession(int modelID) {
        try {
            ortEnv = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
//             NNAPI: Android 8.1 (API 27) or higher
            int androidSdkVer = Build.VERSION.SDK_INT;
            if (androidSdkVer >= 27) {

                EnumSet<NNAPIFlags> flags = EnumSet.noneOf(NNAPIFlags.class);
                flags.add(NNAPIFlags.USE_FP16);
//                if (androidSdkVer >= 29) {
//                    flags.add(NNAPIFlags.CPU_DISABLED);
//                }
                sessionOptions.addNnapi(flags);
            }
            ortSession = ortEnv.createSession(ByteStreamsKt.readBytes(resources.openRawResource(modelID)), sessionOptions);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void readClasses(int classesFileID) {
        InputStream inputStream = resources.openRawResource(classesFileID);
        // read lines
        HashMap<Integer, String> readClasses = new HashMap<>();
        int i = 0;
        try (java.util.Scanner scanner = new java.util.Scanner(inputStream)) {
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                readClasses.put(i, line);
                i++;
            }
        }
        classMapping = readClasses;
    }

    private static class PreprocessedImage {
        public FloatBuffer imageData;
        public int padX;
        public int padY;

        public PreprocessedImage(FloatBuffer imageData, int padX, int padY) {
            this.imageData = imageData;
            this.padX = padX;
            this.padY = padY;
        }
    }

    private PreprocessedImage preprocess(Bitmap image) {
        // Resize
        Bitmap resizedBm = ImageUtils.resizeKeepRatio(image, inferSize);

        // Pad
        ImageUtils.PaddedImage paddedImage = ImageUtils.pad(resizedBm, inferSize, PAD_VAL);
        Bitmap paddedBm = paddedImage.image;
        int padX = paddedImage.padX;
        int padY = paddedImage.padY;

        // Convert to float array
        FloatBuffer imageData = ImageUtils.normalizeImage(paddedBm, MEAN, STD);

        return new PreprocessedImage(imageData, padX, padY);
    }


    public DetectionResult infer(Bitmap inputBitmap) throws OrtException {
        long startTime = 0L;
        long endTime = 0L;
        long totalTime = 0L;

        int origWidth = inputBitmap.getWidth();
        int origHeight = inputBitmap.getHeight();

        ////////////////////////////////////////
        // Preprocessing
        startTime = System.currentTimeMillis();

        PreprocessedImage preprocessedImage = preprocess(inputBitmap);
        FloatBuffer inputData = preprocessedImage.imageData;
        int padX = preprocessedImage.padX;
        int padY = preprocessedImage.padY;

        // Input collections
        OnnxTensor inputTensor = OnnxTensor.createTensor(ortEnv, inputData, new long[]{1, 3, inferSize, inferSize});
        String inputName = ortSession.getInputNames().iterator().next();
        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put(inputName, inputTensor);

        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 1. Pre-process time: " + (endTime - startTime) + "ms");


        ////////////////////////////////////////
        // Inference

        startTime = System.currentTimeMillis();
        // Run ONNX session
        OrtSession.Result output = ortSession.run(inputMap);
        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 2. Inference time: " + (endTime - startTime) + "ms");

        ////////////////////////////////////////
        // Postprocessing

        startTime = System.currentTimeMillis();

        // Extract results from session
        long[] labels = ((long[][]) ((OnnxTensor) output.get(1)).getValue())[0];  // in shape (n)
        float[][] dets = ((float[][][]) ((OnnxTensor) output.get(0)).getValue())[0];  // in shape (n, 5) - [x1, y1, x2, y2, score]
        int[][] boxes = new int[dets.length][4];     // in shape (n, 4) - [x1, y1, x2, y2]
        float[] scores = new float[dets.length];    // in shape (n)
        for (int i = 0; i < dets.length; i++) {
            boxes[i] = new int[]{(int) dets[i][0], (int) dets[i][1], (int) dets[i][2], (int) dets[i][3]};
            scores[i] = dets[i][4];
        }

        float[][][] masks = ((float[][][][]) ((OnnxTensor) output.get(2)).getValue())[0];      // in shape (n, h, w)

        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 3. Extract result time: " + (endTime - startTime) + "ms");

        startTime = System.currentTimeMillis();
        DetectionResult result = postprocess(boxes, scores, labels, masks, origWidth, origHeight, padX, padY);
        endTime = System.currentTimeMillis();

        totalTime += (endTime - startTime);
        System.out.println("[LOG] 4. Post-process time: " + (endTime - startTime) + "ms");

        System.out.println("[LOG] Total time: " + totalTime + "ms");

        return result;
    }

    private DetectionResult postprocess(int[][] boxes, float[] scores, long[] labels, float[][][] masks, int origWidth, int origHeight, int padX, int padY) {
        int n = boxes.length;
        boolean[] isSkipped = new boolean[n];

        // 1. Filter our low score boxes
        for (int i = 0; i < n; i++) {
            if (scores[i] >= commonThres || (labels[i] == 0 && scores[i] >= personThres))
                continue;
            isSkipped[i] = true;
        }

        // 2. Normalize box coordinates (between 0 and infer size - 1)
        for (int i = 0; i < n; i++) {
            if (isSkipped[i])
                continue;

            int x1 = boxes[i][0];
            int y1 = boxes[i][1];
            int x2 = boxes[i][2];
            int y2 = boxes[i][3];

            if (x1 >= x2 || y1 >= y2) {
                isSkipped[i] = true;
                continue;
            }

            x1 = Math.min(Math.max(padX, x1), inferSize - 1 - padX);
            y1 = Math.min(Math.max(padY, y1), inferSize - 1 - padY);
            x2 = Math.min(Math.max(padX, x2), inferSize - 1 - padX);
            y2 = Math.min(Math.max(padY, y2), inferSize - 1 - padY);

            boxes[i][0] = x1; boxes[i][1] = y1; boxes[i][2] = x2; boxes[i][3] = y2;
        }

        // 3. Reduce redundant boxes: NMS + Merged overlapping boxes
        HashMap<Integer, ArrayList<Integer>> mergeDict = new HashMap<>();
        for (int i = 0; i < n; i++) {
            if (isSkipped[i]) {
                continue;
            }
            if (!mergeDict.containsKey(i)) {
                mergeDict.put(i, new ArrayList<Integer>());
            }
            int[] box1 = boxes[i];

            for (int j = i + 1; j < n; j++) {
                if (isSkipped[j]) {
                    continue;
                }
                if (!mergeDict.containsKey(j)) {
                    mergeDict.put(j, new ArrayList<Integer>());
                }
                int[] box2 = boxes[j];

                float boxIoU = calcBoxIoU(box1, box2);

                // crop 2 masks to the same shape
                int x1 = Math.min(box1[0], box2[0]);
                int y1 = Math.min(box1[1], box2[1]);
                int x2 = Math.max(box1[2], box2[2]);
                int y2 = Math.max(box1[3], box2[3]);
                byte[] mask1Crop = cropMask(masks[i], x1, y1, x2, y2);
                byte[] mask2Crop = cropMask(masks[j], x1, y1, x2, y2);
                // calculate mask IoU and overlap
                float maskInter = 0, mask1Area = 0, mask2Area = 0;
                for (int k = 0; k < mask1Crop.length; k++) {
                    if ((mask1Crop[k] & mask2Crop[k]) == 1)
                        maskInter++;
                    if (mask1Crop[k] == 1)
                        mask1Area++;
                    if (mask2Crop[k] == 1)
                        mask2Area++;
                }
                float maskUnion = mask1Area + mask2Area - maskInter + EPS;
                float mask1Overlap = (float) (maskInter / ((float) mask1Area + EPS));
                float mask2Overlap = (float) (maskInter / ((float) mask2Area + EPS));

                // check condition
                if ((boxIoU > BOX_IOU_THRES && (float) maskInter / ((float) maskUnion + 1e-6) > MASK_IOU_THRES) ||
                        (labels[i] == labels[j] && (Math.max(mask1Overlap, mask2Overlap) > OVERLAP_THRES))) {
                    if (scores[i] > scores[j]) {
                        isSkipped[j] = true;
                        mergeDict.get(i).add(j); mergeDict.get(i).addAll(mergeDict.get(j));
                        mergeDict.remove(j);
                    } else {
                        isSkipped[i] = true;
                        mergeDict.get(j).add(i); mergeDict.get(j).addAll(mergeDict.get(i));
                        mergeDict.remove(i);
                    }
                }

                if (isSkipped[i]) {
                    break;
                }
            }
        }

        // 4. Merge masks
        for (int i = 0; i < n; i++) {
            if (isSkipped[i] || !mergeDict.containsKey(i)) {
                continue;
            }

            int[] curBox = boxes[i];
            float[][] curMask = masks[i];

            for (int j = 0; j < mergeDict.get(i).size(); j++) {
                int idx = mergeDict.get(i).get(j);
                int[] box2 = boxes[idx];
                float[][] mask2 = masks[idx];

                // merge box
                curBox[0] = Math.min(curBox[0], box2[0]);
                curBox[1] = Math.min(curBox[1], box2[1]);
                curBox[2] = Math.max(curBox[2], box2[2]);
                curBox[3] = Math.max(curBox[3], box2[3]);
                // merge mask
                for (int k = box2[1]; k <= box2[3]; k++) {
                    for (int l = box2[0]; l <= box2[2]; l++) {
                        curMask[k][l] = Math.max(curMask[k][l], mask2[k][l]);
                    }
                }
            }

            boxes[i] = curBox;
            masks[i] = curMask;
        }
        
        // 5. Refine boxes coordinates
        ArrayList<Float> finalScores = new ArrayList<>();
        ArrayList<int[]> finalBoxes = new ArrayList<>();
        ArrayList<String> finalLabels = new ArrayList<>();
        ArrayList<Bitmap> finalMasks = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            if (isSkipped[i]) {
                continue;
            }
            
            // actual box coordinates
            int x1 = boxes[i][0];
            int y1 = boxes[i][1];
            int x2 = boxes[i][2];
            int y2 = boxes[i][3];
            int actualX1 = (int) ((x1 - padX) / (float) (inferSize - padX * 2) * origWidth);
            int actualY1 = (int) ((y1 - padY) / (float) (inferSize - padY * 2) * origHeight);
            int actualX2 = (int) ((x2 - padX) / (float) (inferSize - padX * 2) * origWidth);
            int actualY2 = (int) ((y2 - padY) / (float) (inferSize - padY * 2) * origHeight);
            // check box size
            if ((actualX2 - actualX1 + 1) + (actualY2 - actualY1 + 1) < BOX_THRES)
                continue;

            // crop current mask (H x W) to final mask (same size with box)
            int maskHeight = y2 - y1;
            int maskWidth = x2 - x1;
            Bitmap maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888);
            int[] binValues = new int[maskWidth * maskHeight];
            int idx = 0;
            for (int j = 0; j < maskHeight; j++) {
                for (int k = 0; k < maskWidth; k++) {
                    int val = Math.round(masks[i][y1 + j][x1 + k]);
                    binValues[idx++] = Color.rgb(val, val, val);
                }
            }
            int maskNewWidth = actualX2 - actualX1;
            int maskNewHeight = actualY2 - actualY1;
            maskBitmap.setPixels(binValues, 0, maskWidth, 0, 0, maskWidth, maskHeight);
            Bitmap actualMaskBitmap = Bitmap.createScaledBitmap(maskBitmap, maskNewWidth, maskNewHeight, false);

            finalBoxes.add(new int[]{actualX1, actualY1, actualX2, actualY2});
            finalMasks.add(actualMaskBitmap);
            finalScores.add(scores[i]);
            finalLabels.add(classMapping.get((int)labels[i]));
        }
        
        return new DetectionResult(finalBoxes, finalMasks, finalScores, finalLabels);
    }

    private float calcBoxIoU(int[] box1, int[] box2) {
        int x1 = Math.max(box1[0], box2[0]);
        int y1 = Math.max(box1[1], box2[1]);
        int x2 = Math.min(box1[2], box2[2]);
        int y2 = Math.min(box1[3], box2[3]);
        float inter = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        if (inter == 0) {
            return 0;
        }
        float area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1);
        float area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1);
        return inter / (area1 + area2 - inter);
    }

//    private byte[] cropMask(byte[][] mask, int x1, int y1, int x2, int y2) {
    private byte[] cropMask(float[][] mask, int x1, int y1, int x2, int y2) {
        int maskHeight = y2 - y1;
        int maskWidth = x2 - x1;
        byte[] maskCrop = new byte[maskHeight * maskWidth];
        int idx = 0;
        for (int i = 0; i < maskHeight; i++) {
            for (int j = 0; j < maskWidth; j++) {
                maskCrop[idx++] = (byte) Math.round(mask[y1 + i][x1 + j]);
            }
        }
        return maskCrop;
    }
}
