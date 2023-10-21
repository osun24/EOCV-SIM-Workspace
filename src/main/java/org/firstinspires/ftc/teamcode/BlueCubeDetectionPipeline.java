public class BlueCubeDetectionPipeline extends OpenCvPipeline {
    private Telemetry telemetry;  // Add Telemetry object
    Mat hsv = new Mat();
    Mat mask = new Mat();
    Mat morphed = new Mat();

    public BlueCubeDetectionPipeline(Telemetry telemetry) {  // Constructor to initialize telemetry
        this.telemetry = telemetry;
    }

    @Override
    public Mat processFrame(Mat input) {
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        Scalar lowerBlue = new Scalar(90, 100, 100);
        Scalar upperBlue = new Scalar(140, 255, 255);
        Core.inRange(hsv, lowerBlue, upperBlue, mask);
        
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
        Imgproc.morphologyEx(mask, morphed, Imgproc.MORPH_OPEN, kernel);
        Imgproc.morphologyEx(morphed, morphed, Imgproc.MORPH_CLOSE, kernel);
        
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(morphed, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        for (MatOfPoint contour : contours) {
            Rect rect = Imgproc.boundingRect(contour);
            double centerX = rect.x + rect.width / 2.0;
            double centerY = rect.y + rect.height / 2.0;

            // Telemetry data for the center coordinates of each blue object
            telemetry.addData("CenterX", centerX);
            telemetry.addData("CenterY", centerY);

            // Telemetry data for the area of each blue object
            telemetry.addData("Area", Imgproc.contourArea(contour));

            // adjust for location identification
            if (centerX < input.cols() / 3) {
                telemetry.addData("Position", "Left");
            } else if (centerX < 2 * input.cols() / 3) {
                telemetry.addData("Position", "Center");
            } else {
                telemetry.addData("Position", "Right");
            }

            Imgproc.rectangle(input, rect.tl(), rect.br(), new Scalar(0, 255, 0), 2);
        }

        telemetry.update();  // Ensure you update telemetry after adding data
        return input;
    }
}
