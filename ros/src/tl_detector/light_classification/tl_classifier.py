import cv2
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

def load_graph(graph_file):
    config = tf.ConfigProto()

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        return sess, ops

class TLClassifier(object):
    def __init__(self, real_world):
        # load classifier
        if real_world:
            sess, _ = load_graph('models/real_model.pb')
        else:
            sess, _ = load_graph('models/sim_model.pb')

        self.sess = sess
        # Define input and output tensors for session
        self.image_tensor = self.sess.graph.get_tensor_by_name('image_tensor:0')
        self.detect_boxes = self.sess.graph.get_tensor_by_name('detection_boxes:0')
        self.detect_scores = self.sess.graph.get_tensor_by_name('detection_scores:0')
        self.detect_classes = self.sess.graph.get_tensor_by_name('detection_classes:0')
        self.num_classes = 3
        self.categories = {
            1: {'id': 1, 'name': u'red'},
            2: {'id': 2, 'name': u'yellow'},
            3: {'id': 3, 'name': u'green'}}

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0)

        (boxes, scores, classes) = self.sess.run(
            [self.detect_boxes, self.detect_scores, self.detect_classes], feed_dict={self.image_tensor: image})

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        prediction = 4
        min_score_thresh = .75

        for i in range(boxes.shape[0]):
            if (scores is None or scores[i] > min_score_thresh) and classes[i] in [1,2,3]:
                prediction = classes[i]
                min_score_thresh = scores[i]
                print("Traffic light: %s - %.4f" % (str(self.categories[classes[i]]['name']), scores[i]))               

        light_prediction = TrafficLight.UNKNOWN
        if prediction == 1:
            light_prediction = TrafficLight.RED
        elif prediction == 2:
            light_prediction = TrafficLight.YELLOW
        elif prediction == 3:
            light_prediction = TrafficLight.GREEN

        return light_prediction
