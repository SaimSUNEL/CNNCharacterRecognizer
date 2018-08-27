import tensorflow as tf
import MyImageLoader
from cv2 import *
import numpy as np
import cv2



x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, len(MyImageLoader.Y[0])])

input_ = 3 * 3 * 1
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W = tf.get_variable("W", (3, 3, 1, 64), tf.float32, initializer)
b = tf.get_variable("b", [64], tf.float32, tf.constant_initializer(0))

conv1 = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

conv1_out = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b))
max_pool = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

input_ = 3 * 3 * 64
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W2 = tf.get_variable("W2", (3, 3, 64, 32), tf.float32, initializer)
b2 = tf.get_variable("b2", [32], tf.float32, tf.constant_initializer(0))

conv2 = tf.nn.conv2d(max_pool, W2, strides=[1, 3, 3, 1], padding="SAME")

conv2_out = tf.nn.leaky_relu(tf.nn.bias_add(conv2, b2))
max_pool2 = tf.nn.max_pool(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

conv2_normalize = tf.reshape(max_pool2, shape=[-1, 3 * 3 * 32])
input_ = 3 * 3 * 32
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W3 = tf.get_variable("W3", (3 * 3 * 32, 500), tf.float32, initializer)
b3 = tf.get_variable("b3", [500], tf.float32, tf.constant_initializer(0))

f1_output = tf.nn.tanh(tf.matmul(conv2_normalize, W3) + b3)

input_ = 500
initializer = tf.random_normal_initializer(stddev=(2.0 / input_) ** 0.5)
W4 = tf.get_variable("W4", (500, len(MyImageLoader.Y[0])), tf.float32, initializer)
b4 = tf.get_variable("b4", [len(MyImageLoader.Y[0])], tf.float32, tf.constant_initializer(0))

f2_output = tf.nn.softmax(tf.nn.leaky_relu(tf.matmul(f1_output, W4) + b4))
global_step = tf.Variable(0, name="global_step", trainable=False)


entropy = y * tf.log(f2_output)
cross_entropy = -tf.reduce_sum(entropy)
loss = tf.reduce_sum(cross_entropy)


train_saver = tf.train.Saver()
session = tf.Session()

train_saver.restore(session, "my-CNN-test-model/my-CNN-test-model")
print "network dosyadan yuklendi..."

def fnc(x1, x2):
    print "x1 : " , x1 , " - x2 : " , x2

    if ( x1[ 0 ] < x2[ 0 ] ):
        return -1

    elif x1[ 0 ] == x2[ 0 ] :
        return 0

    else:
        return 1

renkler = { 0: ( 255 , 0 , 0 ) , 1: (0 , 255 , 0) , 2 : ( 0,0,255) , 3: ( 255 ,255 , 0) , 4:( 255 ,0,255 ) , 5: (0 , 255 ,255) , 6 :( 255 , 255 ,255 ) }

numbers = { "sifir" : 0 , "bir" : 1 , "iki":2 , "uc":3 , "dort":4 , "bes":5 , "alti":6 , "yedi":7  , "sekiz" : 8 , "dokuz":9 , "A":'A' , "B":'B' , "C":"C" , "D":'D' , "E":'E' , "F":'F' , "G":'G' , 'H':'H' , "I":'I' , "K":'K' , "M":"M","N":"N","P":"P","S":"S","T":"T","V":"V","Y":"Y","Z":"Z" , "R":"R" , "CH":"CH" , "SH":"SH" , "U":"U" , "J":"J" , "L":"L" }


resim_directory = "CharacterCreater/"


rakamlar = [ "sifir" , "bir" , "iki" , "uc" , "dort" , "bes" , "alti","yedi" , "sekiz" , "dokuz" , "A" , "B" , "C" , "D" , "E" , "F"  , "G" , "H" , "I" , "K","M","N","P","S","T","V","Y","Z" , "R" , "CH"  , "SH"  , "U" , "J" , "L" ]


def mouse_fonksiyonu ( *param):
    global character_image

    if ( param [ 3 ] == 1 ):
        #x , y terslendi...
        character_image [ param [ 2 ] - 3 : param[ 2 ] +3 , param [ 1 ] -3 : param[ 1 ] + 3] = 255
        imshow ( "Bos" , character_image )


   # print "Parameters : " , param


first_layer_weights = session.run(W)
print "W : ", first_layer_weights[:,:,0,1]
print("len w : ", len(first_layer_weights))
print("shape w : ", first_layer_weights.shape)

for layer_index in range(0, 64):
    resim = first_layer_weights[:, :, 0, layer_index].copy()
    normalized = cv2.normalize(resim, 0, 1, cv2.NORM_MINMAX )
    cv2.imwrite("weight_images/layer"+str(layer_index)+".png", normalized*255.0)

cv2.imshow("weight", normalized*255.0)
resized = None

character_image = np.zeros ( (480 , 640 , 1 ) , np.uint8 )

follow_image = np.zeros ( ( 480 , 640  , 3 ) , np.uint8 )

control_image = np.zeros ( ( 480 , 640 ,  3 ) , np.uint8 )

palet_height, palet_width, _ = character_image.shape


namedWindow( "Bos" , WINDOW_AUTOSIZE )
namedWindow("Resized" , WINDOW_AUTOSIZE )
imshow ( "Bos" , character_image )

setMouseCallback( "Bos" , mouse_fonksiyonu , None )

previous_character_count = 0

tus = ""
while tus != 27:
    tus = waitKey( 0 )
    #cizilecek resmi resetliyoruz...
    if ( tus == ord ('r')):
        character_image [ : ] = 0
        imshow ( "Bos" , character_image )
    #karakter ismini giriyoruz...
    if tus == ord ( 'p' ):
        son_result = ""

        for image in bulunan_karakter:

          if image is None:
              son_result += "\n"
              continue
          temp = []
          temp.append(1.0)


          for a in range(0, 28):
              for b in range(0, 28):

                  if ( image [ a, b ] . all ( ) > 0  ):
                      temp.append(1)
                  else:
                      temp.append(0)

          res = session.run(f2_output,feed_dict={x: [np.array(temp[1:],dtype=np.float32).reshape(28,28,1) ]} )

          
          print "Result", res
          #res = session.run(f2_output,feed_dict={x:  [temp[1:]] } )

          son_result += str(numbers[rakamlar[np.argmax(res)]])

        print "Son sonuc : " , son_result

    if tus == ord ( 'k' ):
        follow_image[:] = character_image[:]

        clone = character_image.copy()
        res , contours, hier = findContours(character_image, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        character_image = clone

        if (previous_character_count != 0):
            for g in range(0, previous_character_count):
                pass
                #To destroy the appeared windows of contours of the characters...
                #destroyWindow("Resized" + str(g))
            previous_character_count = 0

        # print "Contours" , contours
        # print "Hier : " , hier
        # print "Lem hier : " , len ( contours ),
        bulunan_karakter = []
        noktalar = []

        for _i in range(0, len(contours)):
            x_, y_, w, h = boundingRect(contours[_i])

            noktalar.append((x_, y_, w, h))

        for (x_, y_, w, h) in noktalar:
            control_image[y_:y_ + h, 0:palet_width] = 255

        araliklar = []

        satirlar = []

        print "image : ", control_image[3, 2, 0]

        aralik_basladi = False
        for y in range(0, palet_height-1):
            if (int(control_image[y, 2, 0]) - int(control_image[y + 1, 2, 0])) != 0:
                if aralik_basladi is False:
                    temp = []
                    temp.append(y)
                    aralik_basladi = True

                else:
                    aralik_basladi = False
                    temp.append(y)
                    araliklar.append(temp)

        print "Araliklar : ", araliklar

        for aralik in araliklar:
            satir = []
            for (x_, y_, w, h) in noktalar:
                if aralik[0] <= y_ <= aralik[1]:
                    satir.append((x_, y_, w, h))

            satir.sort(fnc)

            satirlar.append(satir)

        rakam = 0

        index = 0

        for satir in satirlar:

            for (x_, y_, w, h) in satir:
                rectangle(follow_image, (x_, y_), (x_ + w, y_ + h), renkler[2])
                circle(follow_image, (x_, y_), 3, renkler[2])

                char_res = character_image[y_:y_ + h, x_:x_ + w]
                resized = resize(char_res, (28, 28))
                #To show the character contours...
                #imshow("Resized" + str(index), resized)
                bulunan_karakter.append(resized);
                index += 1
                previous_character_count += 1

            rakam += 1
            bulunan_karakter.append ( None )
            print "Satir ", rakam, " : ", satir
            imshow("Follow", follow_image)



            _i = 0