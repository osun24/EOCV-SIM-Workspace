package org.firstinspires.ftc.teamcode;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.opencv.core.*;
import org.openftc.easyopencv.OpenCvPipeline;
import org.opencv.imgproc.Imgproc;
import java.util.ArrayList;
import java.util.List;

public class TrussDetectionPipeline extends OpenCvPipeline {
    private Telemetry telemetry;  // Add Telemetry object
    Mat processedImage = new Mat();

    public TrussDetectionPipeline(Telemetry telemetry) {  // Constructor to initialize telemetry
        this.telemetry = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        // Convert to grayscale
        Imgproc.cvtColor(input, processedImage, Imgproc.COLOR_BGR2GRAY);
        
        // Blur the image
        Imgproc.GaussianBlur(processedImage, processedImage, new Size(5,5), 0);
        
        // Edge detection
        Imgproc.Canny(processedImage, processedImage, 50, 150);
        
        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(processedImage, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        
        // Filter and draw rectangles for vertical columns
        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            double aspect_ratio = (double)rect.width / rect.height;
            
            if (0.1 < aspect_ratio && aspect_ratio < 0.4) {
                Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
            }
        }
        
        return input;
    }
}