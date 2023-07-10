package com.example.rtmdet_ins_snapedit;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.Color;

import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import ai.onnxruntime.extensions.OrtxPackage;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import kotlin.io.ByteStreamsKt;

public class ObjectDetector {
    // constant of current model family
    private static final int PAD_VAL = 114;
    private static final float[] MEAN = {103.53F, 116.28F, 123.675F};
    private static final float[] STD = {57.375F, 57.12F, 58.395F};

    // constant of post-processing
    private static final int BOX_THRES = 20;    // ignore too small boxes


    static class DetectionResult {
        public ArrayList<int[]> boxes;      // (n, 4) - format x1, y1, x2, y2
        public ArrayList<Bitmap> masks;    // (n,) - bitmap of mask corresponding to box
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
    private HashMap<Integer, String> classes;
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
            sessionOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
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
        classes = readClasses;
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

        System.out.println("[LOG] Preprocessing time: " + (endTime - startTime) + "ms");

        ////////////////////////////////////////
        // Inference

        startTime = System.currentTimeMillis();

        OrtSession.Result output = ortSession.run(inputMap);
        OnnxTensor detsTensor = (OnnxTensor) output.get(0);
        OnnxTensor labelsTensor = (OnnxTensor) output.get(1);
        OnnxTensor masksTensor = (OnnxTensor) output.get(2);

        float[][] dets = ((float[][][]) detsTensor.getValue())[0];          // in shape (n, 5) - [x1, y1, x2, y2, score]
        long[] labels = ((long[][]) labelsTensor.getValue())[0];            // in shape (n,)
        float[][][] masks = ((float[][][][]) masksTensor.getValue())[0];    // in shape (n, h, w)

        endTime = System.currentTimeMillis();

        System.out.println("[LOG] Inference time: " + (endTime - startTime) + "ms");

        ////////////////////////////////////////
        // Postprocessing

        startTime = System.currentTimeMillis();

        ArrayList<Float> finalScores = new ArrayList<>();
        ArrayList<int[]> finalBoxes = new ArrayList<>();
        ArrayList<String> finalLabels = new ArrayList<>();
        ArrayList<Bitmap> finalMasks = new ArrayList<>();

        for (int i = 0; i < dets.length; i++) {
            float score = dets[i][4];
            if (score >= commonThres || (labels[i] == 0 && score >= personThres)) {
                int x1 = Math.min(Math.max((int)dets[i][0], padX), inferSize - 1 - padX);
                int y1 = Math.min(Math.max((int)dets[i][1], padY), inferSize - 1 - padY);
                int x2 = Math.min(Math.max((int)dets[i][2], padX), inferSize - 1 - padX);
                int y2 = Math.min(Math.max((int)dets[i][3], padY), inferSize - 1 - padY);
                int actualX1 = (int) ((x1 - padX) / (float) (inferSize - padX * 2) * origWidth);
                int actualY1 = (int) ((y1 - padY) / (float) (inferSize - padY * 2) * origHeight);
                int actualX2 = (int) ((x2 - padX) / (float) (inferSize - padX * 2) * origWidth);
                int actualY2 = (int) ((y2 - padY) / (float) (inferSize - padY * 2) * origHeight);
                // check box size
                if ((actualX2 - actualX1 + 1) + (actualY2 - actualY1 + 1) < BOX_THRES)
                    continue;

                // crop current mask --> actual mask
                int maskHeight = y2 - y1;
                int maskWidth = x2 - x1;
                Bitmap maskBitmap = Bitmap.createBitmap(maskWidth, maskHeight, Bitmap.Config.ARGB_8888);
                int[] binValues = new int[maskWidth * maskHeight];
                int idx = 0;
                for (int j = 0; j < maskHeight; j++) {
                    for (int k = 0; k < maskWidth; k++) {
                        int val = Math.round(masks[i][y1 + j][x1 + k]);
                        binValues[idx++] = Color.rgb(1 * val, 1 * val, 1 * val);
                    }
                }

                int maskNewWidth = actualX2 - actualX1;
                int maskNewHeight = actualY2 - actualY1;
                maskBitmap.setPixels(binValues, 0, maskWidth, 0, 0, maskWidth, maskHeight);
                Bitmap actualMaskBitmap = Bitmap.createScaledBitmap(maskBitmap, maskNewWidth, maskNewHeight, false);

                finalBoxes.add(new int[]{actualX1, actualY1, actualX2, actualY2});
                finalMasks.add(actualMaskBitmap);
                finalScores.add(score);
                finalLabels.add(classes.get((int) labels[i]));
            }
        }

        endTime = System.currentTimeMillis();

        System.out.println("[LOG] Postprocessing time: " + (endTime - startTime) + "ms");

        return new DetectionResult(finalBoxes, finalMasks, finalScores, finalLabels);
    }
}
