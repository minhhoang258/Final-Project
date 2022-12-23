#A. Liệt kê các thư viện để sử dụng

# Các thư viện được dùng để train data
import keras
import numpy as np
from PIL import Image, ImageFile
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Các thư viện được dùng để làm giao diện + tính năng phần mềm
import copy
import cv2
from keras.models import load_model
import time

# Thêm thư viện tkinter để làm giao diện ngoài

from tkinter import *
from PIL import ImageTk, Image

'''

#B. TRAIN DATA

# Đặt tên cử chỉ dựa vào vào những chữ cái đầu tên của ảnh
cu_chi = {'L_': 'I want to advance in life',
           'fi': 'I will try harder!',
           'ok': 'Everything is OK',
           'pe': 'Hi! Have a peaceful day!',
           'pa': 'Please, stop!'}

# Gán các biến tương ứng với đầu ra
cu_chi_map = {'I will try harder!': 0,
                'I want to advance in life': 1,
                'Everything is OK': 2,
                'Hi! Have a peaceful day!': 3,
                'Please, stop!': 4}

cu_chi_names = {0: 'I will try harder!',
                 1: 'I want to advance in life',
                 2: 'Everything is OK',
                 3: 'Hi! Have a peaceful day!',
                 4: 'Please, stop!'}

# Gán các biến truy cập vào vào đường dẫn
anh_path = '/content/drive/MyDrive/AI_project'
models_path = 'models/File_train.hdf5'
rgb = False
Sizeanh = 224

#1. TẠO HÀM XỬ LÝ DATA

# Tạo hàm xử lí ảnh resize vè 224x224 và chuyển về numpy array
def xu_li_anh(path):
    img = Image.open(path)
    img = img.resize((Sizeanh, Sizeanh))
    img = np.array(img)
    return img

# Tạo hàm xử lí dữ liệu đầu vào
def xu_li_du_lieu(X_data, y_data):
    X_data = np.array(X_data, dtype = 'float32')
    if rgb:
        pass
    else:
        X_data = np.stack((X_data,)*3, axis=-1)
    X_data /= 255
    y_data = np.array(y_data)
    y_data = to_categorical(y_data)
    return X_data, y_data

# Tạo hàm duyệt thư mục ảnh dùng để train
def duyetdata(anh_path):
    X_data = []
    y_data = []
    for directory, subdirectories, files in os.walk(anh_path):
        for file in files:
            if not file.startswith('.'):
                path = os.path.join(directory, file)
                cu_chi_names = cu_chi[file[0:2]]
                print(cu_chi_names)
                print(cu_chi_map[cu_chi_names])
                y_data.append(cu_chi_map[cu_chi_names])
                X_data.append(xu_li_anh(path))

            else:
                continue

    X_data, y_data = xu_li_du_lieu(X_data, y_data)
    return X_data, y_data

#2 XỬ LÝ DATA VÀ TRAIN DATA

# Load dữ liệu vào X, Y đồng thời phân chia dữ liệu theo tỉ lệ train:test là 70:30
X_data, y_data = duyetdata(anh_path)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size = 0.3, random_state=12, stratify=y_data)

# Đặt các checkpoint để lưu lại model tốt nhất

model_checkpoint = ModelCheckpoint(filepath=models_path, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_acc',
                               min_delta=0,
                               patience=10,
                               verbose=1,
                               mode='auto',
                               restore_best_weights=True)

# Khởi tạo model
model1 = VGG16(weights='imagenet', include_top=False, input_shape=(Sizeanh, Sizeanh, 3))
optimizer1 = optimizers.Adam()
base_model = model1

# Duỗi thẳng dữ liệu
a = base_model.output
a = Flatten()(x)

# Cho dữ liệu đi qua 4 lớp Hiden
a = Dense(128, activation='relu', name='fc1')(a)
a = Dense(128, activation='relu', name='fc2')(a)
a = Dense(128, activation='relu', name='fc2a')(a)
a = Dense(128, activation='relu', name='fc3')(a)

# Đưa dữ liệu đi qua lớp ấn với 64 tín hiệu ra
a = Dropout(0.5)(a)
a = Dense(64, activation='relu', name='fc4')(a)

# Cuối cùng đi qua lớp softmax với 5 tín hiệu ra sau cùng
predictions = Dense(5, activation='softmax')(a)
model = Model(inputs=base_model.input, outputs=predictions)

# Chọn chỉ train 5 tín hiệu ra sau cùng
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=64, validation_data=(X_test, y_test), verbose=1, 
    callbacks=[early_stopping, model_checkpoint])

# Lưu lại Model đã train

model.save('/content/drive/MyDrive/File_train/Data_train.h5')'''

#C. Thiết kế giao diện + tính năng phần mềm

# Tạo giao diện
class Giao_dien_ban_dau:    

# Tạo cửa sổ menu
    def __init__(self,root):
        self.root = root
        self.root.title("TRANSLATION SOFTWARE FOR THE DEAF")
        self.root.geometry("1080x600+200+80")
        self.root.resizable(0,0)

# Trang trí cửa sổ menu

    # Trang trí giao diện
        img1=Image.open(r"trangtrilogin\bg5.png")
        self.photoimgae1=ImageTk.PhotoImage(img1)
        self.lblimg1=Label(self.root,image=self.photoimgae1,bd=0)
        self.lblimg1.place(x=0,y=0)

        Main_Frame = Canvas(self.root,bg='#f0ffff')
        Main_Frame.place(x=350,y=0,width=340,height=600)
        
        img2=Image.open(r"trangtrilogin\Hinh-Nen-Chuyen-Dong-Cho-Dien-Thoai-Dep-Nhat-2022.GIF")
        self.photoimgae2=ImageTk.PhotoImage(img2)
        self.lblimg2=Label(Main_Frame,image=self.photoimgae2,bg='white')
        self.lblimg2.place(x=-2,y=-0.5)

        nen=Label(Main_Frame,text='WELCOME!',font=('Cooper Black',19),fg='black',bg='#6495ed')
        nen.place(x=105,y=50)

        nen1=Label(Main_Frame,text='AI TRANSLAION SOFTWARE',font=('Agency FB',23,'bold'),fg='black',bg='#6495ed')
        nen1.place(x=34,y=80)

        nen2=Label(Main_Frame,text='Nhấn START để khởi chạy chương trình',font=('Times New Roman',13,'bold'),fg='red',bg='#f0ffff')
        nen2.place(x=30,y=300)

        nen3=Label(Main_Frame,bg='#fffafa',text='Được thực hiện bởi Lê Đình Hoàng Minh - 20104043',font=('Times New Roman',7,'bold'),fg='silver',width=70)
        nen3.place(x=0,y=580)

    # Thêm các nút ấn mang chức năng khác nhau
        lg_button1 = Button(Main_Frame,fg='black',bg='#ffe4e1',border=0 ,text='START',width ='15',height='2',
            font =('Stylus BT',11,'bold'), cursor= "hand2",activeforeground='#d3d3d3',command=self.clickstart).place(x=100,y=340)

        lg_button2 = Button(Main_Frame,fg='black',bg='#ffe4e1',border=0 ,text='How To Use',width ='15',height='2',
            font =('Stylus BT',11,'bold'), cursor= "hand2",activeforeground='#d3d3d3',command=self.huongdansudung).place(x=100,y=400)

        lg_button1 = Button(Main_Frame,fg='black',bg='#ffe4e1',border=0 ,text='Exit',width ='15',height='2',
            font =('Stylus BT',11,'bold'), cursor= "hand2",activeforeground='#d3d3d3',command=self.Exit).place(x=100,y=460)

# Tạo công dụng cho nút truy cập vào chương trình
    def clickstart(self):
    # Hủy cửa sổ menu
        self.root.destroy()
    # Code chạy chương trình        
# Khai báo các biến phục vụ cho việc tạo tính năng nhận diện + dự đoán
        pred = ''
        Accuracy = 0
        bgModel = None

        cu_chi_names = {0: 'I will try harder!',
                    1: 'I want to advance in life',
                    2: 'Everything is OK',
                    3: 'Hi! Have a peaceful day!',
                    4: 'Please, stop!'}

        toa_do_x_begin = 0.5
        toa_do_y_end = 0.6
        threshold = 60
        blurValue = 41
        bgSubThreshold = 50
        learningRate = 0
        predThreshold= 100
        ghi_nhan_Captured = 0

# Lấy model đã train từ trước
        model = load_model("Data_train.h5")

# Tạo hàm trả kết quả dự đoán (đoán hành động đó mang ý nghĩa gì đồng thời hiện thị độ chính xác dựa trên mạng đã train)
        def ketqua(image1):
            image1 = np.array(image1, dtype='float32')
            image1 /= 255
            pred_array = model.predict(image1)
            print(f'pred_array: {pred_array}')
            result = cu_chi_names[np.argmax(pred_array)]
            print(f'Result: {result}')
            print(max(pred_array[0]))
            Accuracy = float("%0.2f" % (max(pred_array[0]) * 100))
            print(result)
            return result, Accuracy

# Tạo hàm tách ảnh khỏi nền
        def tach_nen(frame):
            fgmask = bgModel.apply(frame, learningRate=learningRate)
            kernel = np.ones((3, 3), np.uint8)
            fgmask = cv2.erode(fgmask, kernel, iterations=1)
            res = cv2.bitwise_and(frame, frame, mask=fgmask)
            return res

# Truy cập vào camera
        camera = cv2.VideoCapture(0)
        camera.set(10,200)
        camera.set(cv2.WINDOW_FULLSCREEN,1)

        while camera.isOpened():
# Hiện khung từ webcam
            ret, frame = camera.read()
# Làm mịn khung hình
            frame = cv2.bilateralFilter(frame, 5, 50,100)
            frame = cv2.flip(frame, 1)

# Vẽ khung hình chữ nhật xác định vùng có cử chỉ nhận dạng
            cv2.rectangle(frame, (int(toa_do_x_begin * frame.shape[1]), 0),
                (frame.shape[1], int(toa_do_y_end * frame.shape[0])), (255, 0, 0), 2)

# Nếu đang ghi màn hình thì tách nên dựa vừa hàm vừa tạo trên
            if ghi_nhan_Captured == 1:
                img = tach_nen(frame)

                img = img[0:int(toa_do_y_end * frame.shape[0]),
                int(toa_do_x_begin * frame.shape[1]):frame.shape[1]]

# Tạo 2 cửa sổ khác để nhận diện (mục đích để dự đoán chuẩn hơn bởi ảnh train có cùng tính chất đen trắng)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

                cv2.imshow('GESTURE RECOGNITION', cv2.resize(blur, dsize=None, fx=1, fy=1))

                ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                cv2.imshow('GESTURE RECOGNITION 2', cv2.resize(thresh, dsize=None, fx=1, fy=1))

                if (np.count_nonzero(thresh)/(thresh.shape[0]*thresh.shape[0])>0.2):

# Nếu đã lấy được nét hình bàn tay thì dựa vào mạng đã train để dự đoán
                    if (thresh is not None):
                        target = np.stack((thresh,) * 3, axis=-1)
                        target = cv2.resize(target, (224, 224))
                        target = target.reshape(1, 224, 224, 3)
                        pred, Accuracy = ketqua(target)
                        print(Accuracy,pred)
                        if (Accuracy>=predThreshold):
                            cv2.putText(frame, "Mean: " + pred, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                (50, 250, 0), 4, lineType=cv2.LINE_AA)
            thresh = None

# Thiết lập các tính năng nút bấm
            k = cv2.waitKey(10)

# Bấm P để về menu chính
            if k == ord('p'):
                cv2.destroyAllWindows()
                st_root = Tk()
                st = Giao_dien_ban_dau(st_root)
                st_root.mainloop()
                break

# Bấm E để thoát hoàn toàn ứng dụng:
            if k == ord('e'):
                break

# Bấm phím cách "Space" để ghi màn hình
            elif k == ord(' '):
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)

                ghi_nhan_Captured = 1
                cv2.putText(frame, "Record BG", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 6, lineType=cv2.LINE_AA)
                time.sleep(2)
                print('Mode: Record BG')

            elif k == ord('r'):

                bgModel = None
                ghi_nhan_Captured = 0
                cv2.putText(frame, "Reset", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255),6,lineType=cv2.LINE_AA)
                print('Reset hệ thống')
                time.sleep(0.01)

            cv2.imshow('SIGN LANGUAGE TRANSLATION WITH AI', cv2.resize(frame, dsize=None, fx=2, fy=2))    

        cv2.destroyAllWindows()
        camera.release()

# Tạo hàm chức năng cho nút hướng dẫn
    def huongdansudung(self):
    # Ngưng cửa sổ menu
        self.root.destroy()
    # Tạo 1 cửa số mới dùng để hướng cách dùng chương trình
        self.root_1 = Tk()
        self.root_1.title('Hướng dẫn sử dụng')
        self.root_1.geometry("1080x600+200+80")
    # Trang trí cửa số theo ý
        Main_Frame1 = Canvas(self.root_1,bg='antiquewhite')
        Main_Frame1.place(x=0,y=0,width=1080,height=600+20+80)

        chu1=Label(Main_Frame1,text='TUTORIAL',font=('Times New Roman',35,'bold'),fg='black',bg='aquamarine',width ='39')
        chu1.place(x=0,y=10)

        chu2=Label(Main_Frame1,text='- Bấm phím Space để bật mode nhận diện. Đưa tay vào khung diện để hệ thống đưa ra kết quả ',font=('Times New Roman',15,'bold'),fg='black',bg='antiquewhite')
        chu2.place(x=30,y=210)

        chu3=Label(Main_Frame1,text='- Bấm phím R để làm mới nhận diện (bấm Space để thực hiện lại nhận diện)',font=('Times New Roman',15,'bold'),fg='black',bg='antiquewhite')
        chu3.place(x=30,y=260)

        chu4=Label(Main_Frame1,text='- Bấm phím P để quay về Menu',font=('Times New Roman',15,'bold'),fg='black',bg='antiquewhite')
        chu4.place(x=30,y=310)

        chu4=Label(Main_Frame1,text='- Bấm phím E để thoát hoàn toàn',font=('Times New Roman',15,'bold'),fg='black',bg='antiquewhite')
        chu4.place(x=30,y=360)


        quaylai1 = Button(Main_Frame1,border=0 ,text='> Trở lại!',width ='15',height='2',font =('time new roman',15,'bold'), 
            cursor= "hand2",command=self.quaylaia).place(x=450,y=500)

# Tạo hàm cho nút 'Trở lại!' trong cửa sổ hướng dẫn
    def quaylaia(self):
    # Ngừng chạy của sổ hướng dẫn
        self.root_1.destroy()
    # Chạy lại của số menu
        st_root = Tk()
        st = Giao_dien_ban_dau(st_root)
        st_root.mainloop()

# Tạo hàm cho nút thoát hoàn toàn chương trình
    def Exit(self):
        self.root.destroy()

# Duy trì cửa sổ, hiển thị bắt đầu từ giao diện menu
root = Tk()
st = Giao_dien_ban_dau(root)
root.mainloop()         