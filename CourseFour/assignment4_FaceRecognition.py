from DeepLearning.CourseFour.fr_utils import *
from DeepLearning.CourseFour.inception_blocks_v2 import *

K.set_image_data_format('channels_first')


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)))
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)))
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """
    encoding = img_to_encoding(image_path, model)
    dist = np.linalg.norm(encoding - database[identity])
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False
    return dist, door_open


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image_path, model)
    min_dist = 100
    identity = None
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(encoding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))
    return min_dist, identity



FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

database = {}
database["danielle"] = img_to_encoding("./images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("./images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("./images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("./images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("./images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("./images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("./images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("./images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("./images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("./images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("./images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("./images/arnaud.jpg", FRmodel)


# Face verification
dist1, door_open1 = verify("./images/camera_0.jpg", "younes", database, FRmodel)
dist2, door_open2 = verify("./images/camera_2.jpg", "kian", database, FRmodel)
print(dist1)
print(dist2)

# Face recognition
dist, name = who_is_it("./images/camera_0.jpg", database, FRmodel)
