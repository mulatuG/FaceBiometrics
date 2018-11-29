from django.http import HttpResponse
from django.contrib import messages

from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render, redirect
from .models import Customer, PCCApplication
import logging
from sklearn.model_selection import train_test_split
from . import dataset_fetch as df
from PIL import Image

from time import time
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle
import cv2 as cv2
import numpy as np
from django.conf import settings
# correct way
BASE_DIR = settings.BASE_DIR
from .forms import  CustomerForm2, PCCApplicationForm, CrimeForm
#login required
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
#@login_required(login_url='/accounts/login/')
@login_required()

# Create your views here.

def index(request):
           
    latest_customer_list = Customer.objects.order_by('-date')[:5]
    context = {
        'latest_customer_list': latest_customer_list,
    }
    return render(request, 'FaceBiometricsApp/index.html', context)

def detail(request, passport_number_or_residence_id):
    customer = get_object_or_404(Customer, passport_number_or_residence_id=passport_number_or_residence_id)
    print('recognized customer id: ')
    print(customer.id)
    #get the latest application id
    id=PCCApplication.objects.order_by('-id')[:1]
    print(id)
    application=PCCApplication.objects.get(customer=customer.id)
    #application=PCCAplication.objects.order_by(customer=customer.id)[:1]
    
    context={
           
           'customer' : customer,
           'application' : application,
           #'crime' : crime,
    }
    return render(request, 'FaceBiometricsApp/detail.html', context)

def newCustomer(request):
    latest_customer_list=Customer.objects.latest('id')
    print (latest_customer_list.id)
    context= {
        'latest_customer_list': latest_customer_list
    }
    return render(request, 'FaceBiometricsApp/new_customer.html', context)


def results(request, customer_id):
    response = "You're looking at the results of customer %s."
    return HttpResponse(response % customer_id)

def newCustomerForm(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = CustomerForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            customer = form.save(commit=False)
            customer.save()
        else :
            form=CustomerForm()
            
        return render(request, 'FaceBiometricsApp/new_customer.html',{'form': form})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = CustomerForm()
        return render(request, 'FaceBiometricsApp/new_customer.html',  {'form': form})


def customer_list(request):
    customers= Customer.objects.all()
    context ={
        'customers':customers
    }
    return render(request, 'FaceBiometricsApp/customer_list.html', context)
#customer detail registration logic goes here
def add_customer(request):
    form= CustomerForm2(request.POST or None)
    success=False
    if form.is_valid():
       passport_number_or_residence_id= form.cleaned_data['passport_number_or_residence_id']
       form.save()
       success=True
       #print(passport_number_or_residence_id)
       #sampl photo
       
       BASE_DIR = settings.BASE_DIR
       #print(BASE_DIR)
       face_cascade = cv2.CascadeClassifier(BASE_DIR+'/FaceBiometricsApp/haarcascade_frontalface_default.xml')
       eye_cascade = cv2.CascadeClassifier('FaceBiometricsApp/haarcascade_eye.xml')
       cap = cv2.VideoCapture(0)
       counter=0
       while 1:
           rec,img = cap.read()
           gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           faces = face_cascade.detectMultiScale(gray, 1.3, 5)

           for (x,y,w,h) in faces:
               cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,250),3)
               #compute region of interest 
               roi_gray = gray[y:y+h, x:x+w]
               roi_color = img[y:y+h, x:x+w]
               
               eyes = eye_cascade.detectMultiScale(roi_gray)
               for (ex,ey,ew,eh) in eyes:
                   cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(10,255,20),2)
                   #take samples when eyes are detected to emprove detection accurace  
                   counter=counter + 1 #count taken sample faces
                   cv2.imwrite('FaceBiometricsApp/MachineLearning/Dataset/customer.'+str(passport_number_or_residence_id)+'.'+str(counter)+'.jpg', gray[y:y+h,x:x+w])
                   # waitKey of 100 millisecond
                   cv2.waitKey(250)

           cv2.imshow('Collecting sample faces',img)
           # @params with the millisecond of delay 1
           cv2.waitKey(1)
           #To get out of the loop
           if(counter>39):
            break
       cap.release()
       cv2.destroyAllWindows()
    form=CustomerForm2()
    context= {'form': form,'message':success }
    return render(request, 'FaceBiometricsApp/new_customer.html', context)


#machine trainer logic goes here below
def MachineTrainUsingEigenFaces(request):
    path = BASE_DIR+'/FaceBiometricsApp/MachineLearning/Dataset'
    import pickle
    # Fetching training and testing dataset along with their image resolution(h,w)
    ids, faces, h, w= df.getImagesWithID(path)
    #print ('features'+str(faces.shape[1]))
    # Spliting training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.25, random_state=42)
    #print (">>>>>>>>>>>>>>> "+str(y_test.size))
    n_classes = y_test.size
    #customer= Customers.objects.values_list('name')[:3]
    #Entry.objects.values_list('id', flat=True).order_by('id')
    #result = [tuple(i.values()) for i in customer]
    # return ValuesQuerySet object
    #customer_list=[entry for entry in customer]
    #print(result)
    target_names =ids
    n_components = 15
    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()

    pca = PCA(n_components=n_components,whiten=True).fit(X_train)
    print('Printing PCA',pca)
    print('prinicpal components',pca.components_)
    print('printing experianced variance',pca.explained_variance_)
   
    # printing the trained data in the form of distribution graph
    def drawingGraphsOrginalImage():
        plt.scatter(X_train[:,0],X_train[:,1],alpha=0.99)
        plt.axis('equal')
        plt.title('Principal Component Analysis')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()
        #drow graph
    
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('orginal')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    #drawingGraphsOrginalImage()
    

    print("done in %0.3fs" % (time() - t0))
    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))
    #Here is an example of using PCA as a dimensionality reduction transform:
    pca.fit(X_train)
    #to display individual eigen faces
    #plt.imshow(pca.mean_.reshape(faces.images[0].shape),cmap=plt.cm.bone)
    #fig = plt.figure(figsize=(16, 6))
    #for i in range(30):
        #ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        #ax.imshow(pca.components_[i].reshape(faces.images[0].shape),cmap=plt.cm.bone)


    X_pca = pca.transform(X_train)
    print("original shape:   ", X_train.shape)
    print("transformed shape:", X_pca.shape)


    # printing the trained data in the form of distribution graph
    def drawingGraphsProjectedImage():
        #pca=PCA(0.95)
        plt.scatter(X_pca[:,0],X_pca[:,1],alpha=0.99)
        plt.axis('equal')
        plt.title('Principal Component Analysis after dimentional reduction')
        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.show()
        #drow graph
        pca = PCA().fit(X_pca)
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.title('Projected')
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.show()
    #drawingGraphsProjectedImage()
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("Predicted labels: ",y_pred)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    # Qualitative evaluation of the predictions using matplotlib

    def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.title(titles[i], size=10)
            plt.xticks(())
            plt.yticks(())
    # plot the result of the prediction on a portion of the test set

    #def title(y_pred, y_test, target_names, i):
     #   pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
      #  true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
       # return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

    #prediction_titles = [title(y_pred, y_test, target_names, i)
     #                   for i in range(y_pred.shape[0])]

    #plot_gallery(X_test, prediction_titles, h, w)
    #plt.show()
    
    # plot the gallery of the most significative eigenfaces
    eigenface_titles = ["Face %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)
    plt.show()

    '''
        -- Saving classifier state with pickle
    '''
    #https://www.datacamp.com/community/tutorials/pickle-python-tutorial
    svm_pkl_filename = BASE_DIR+'/FaceBiometricsApp/MachineLearning/Serializer/svm_classifier.pkl'
    # Open the file to save as pkl file w->write b->binary data
    svm_model_pkl_object = open(svm_pkl_filename, 'wb')
    pickle.dump(clf, svm_model_pkl_object)
    # Close the pickle instances
    svm_model_pkl_object.close()


    #https://www.datacamp.com/community/tutorials/pickle-python-tutorial
    pca_pkl_filename = BASE_DIR+'/FaceBiometricsApp/MachineLearning/Serializer/pca_state.pkl'
    # Open the file to save as pkl file
    pca_pkl = open(pca_pkl_filename, 'wb')
    pickle.dump(pca, pca_pkl)
    # Close the pickle instances
    pca_pkl.close()

    #plt.show()

    return redirect('/FaceBiometricsApp')

#face recognition logic goes here below    
def FaceRecognitionLogic(request):
    url=''
    #load models r->read and b->binary data
    svm_pkl_filename =  BASE_DIR+'/FaceBiometricsApp/MachineLearning/Serializer/svm_classifier.pkl'

    svm_model_pkl = open(svm_pkl_filename, 'rb')
    svm_model = pickle.load(svm_model_pkl)
    print ("Loaded SVM model :: ", svm_model)

    pca_pkl_filename =  BASE_DIR+'/FaceBiometricsApp/MachineLearning/Serializer/pca_state.pkl'

    pca_model_pkl = open(pca_pkl_filename, 'rb')
    pca = pickle.load(pca_model_pkl)
    print ('Loaded pca Model :: ', pca)
    facedata = cv2.CascadeClassifier(BASE_DIR+'/FaceBiometricsApp/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('FaceBiometricsApp/haarcascade_eye.xml')
    cascade = facedata
    cap = cv2.VideoCapture(0)
    
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,200,250),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            #now=datetime.now()
            #filenameIndex=now.strftime("%H-%M")
            #print(filenameIndex)
            #import getpass
            #username = getpass.getuser()
            #print (username)
            import socket
            hostname = socket.gethostname()
            #print(hostname)
            fileName=hostname+'_customer.jpg'
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                cv2.imwrite('FaceBiometricsApp/MachineLearning/uploads/'+fileName, gray[y:y+h,x:x+w])
            imgPath=BASE_DIR+'/FaceBiometricsApp/MachineLearning/uploads/'+fileName
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()  
                
    # Converts np array back into image
    img = Image.fromarray(gray)
    # re-sizing to common dimension
    img = img.resize((150,150), Image.ANTIALIAS)
    #img.save('cropped.jpg')
    '''
    Input Image
    '''
    try:
        print('Function calling...')
        #function to normalize the detected image for recognition
        def nomalizingFace(image):
            cascade = cv2.CascadeClassifier(BASE_DIR+'/FaceBiometricsApp/haarcascade_frontalface_default.xml')
            #print('function execution')
            img = cv2.imread(image)
            minisize = (img.shape[1],img.shape[0])
            miniframe = cv2.resize(img, minisize)

            faces = cascade.detectMultiScale(miniframe)

            for f in faces:
                x, y, w, h = [ v for v in f ]
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255))

                sub_face = img[y:y+h, x:x+w]

                #converts img array into grayscale
                gray_image = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                # Converts np array back into image
                img = Image.fromarray(gray_image)
                # re-sizing to common dimension
                img = img.resize((150,150), Image.ANTIALIAS)
                #img.save('cropped.jpg')
                #img.show(img)
                #print('image saved')
            return img


        inputImg =nomalizingFace(imgPath)
        #inputImg.show()
    except :
        print("Customer not Recognized")
        return redirect('/FaceBiometricsApp/error')

    imgNp = np.array(inputImg, 'uint8')
    #Converting 2D array into 1D
    imgFlatten = imgNp.flatten()
    #print imgFlatten
    #print imgNp
    imgArrTwoD = []
    imgArrTwoD.append(imgFlatten)
    # Applyting pca
    img_pca = pca.transform(imgArrTwoD)
    #print img_pca

    pred = svm_model.predict(img_pca)
    print(svm_model.best_estimator_)
    if (str(pred[0])!=''):
        print (pred[0])
        url='/FaceBiometricsApp/detail/'+str(pred[0])
    else:
        print('Face Not Recognized')
        url='/FaceBiometricsApp/error'
    return redirect(url)    
              
def help(request):
           
       return render(request, 'FaceBiometricsApp/help.html')


def pcc_application(request):
    form= PCCApplicationForm(request.POST or None)
    success=False
    if form.is_valid():
       #passport_number_or_residence_id= form.cleaned_data['passport_number_or_residence_id']
       form.save()
       success=True
    form=PCCApplicationForm()
    context= {'form': form,'message':success }
    return render(request, 'FaceBiometricsApp/pcc_application.html', context)
def new_crime(request):
    form=CrimeForm(request.POST or None)
    success=False
    if form.is_valid():
        form.save()
        success=True
    form=CrimeForm()
    context ={'form': form, 'message':success}
    return render(request, 'FaceBiometricsApp/new_crime.html', context)

def error(request):
    return render(request, 'FaceBiometricsApp/404.html', {})