package org.firstinspires.ftc.teamcode;
import android.graphics.Canvas;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.internal.camera.calibration.CameraCalibration;
import org.firstinspires.ftc.vision.VisionProcessor;
import org.opencv.core.*;
import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

// Currently only supports left camera - right camera may need to be added separately...
public class BlueFiltering extends OpenCvPipeline {
    Mat hsv = new Mat();
    Mat mask = new Mat();
    Mat morphed = new Mat();

    public Scalar lowerBlue = new Scalar(50, 40, 50);
    public Scalar upperBlue = new Scalar(130, 190, 190);

    @Override
    public Mat processFrame(Mat frame) {
        Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_RGB2HSV);

        Core.inRange(hsv, lowerBlue, upperBlue, mask);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, morphed, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(morphed, morphed, Imgproc.MORPH_CLOSE, kernel);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            double area = Imgproc.contourArea(contour);

            Rect rect = Imgproc.boundingRect(contour);
            Imgproc.rectangle(morphed, rect.tl(), rect.br(), new Scalar(255, 0, 0), 2);
        }

       return morphed;
    }
}