import cv2
import os
import numpy as np
import heapq 
from sklearn.decomposition import PCA as sklearnPCA

####hyperparameter
COLUMNOFCODEBOOK = 32
DESDIM = 128
SUBVEC = 32
SUBCLUSTER = 256
PCAD = 128
TESTTYPE = 0


def read_file(FILE_PATH = "D://datas//jpg1//jpg//"):
    '''
    Description: read all available images path
    Input: FILE_PATH - given file path
    Output: file_path_list - a list of the available images path
    '''
    cnt = 0
    file_path_list = []
    for picId in range(1000, 1500):
        for eachPic in range(0, 100):
            _file_path = FILE_PATH + str(picId) + str(eachPic).zfill(2) + ".jpg"
            if os.path.exists(_file_path):
                #cv2.imread(_file_path)
                file_path_list.append(_file_path)
                cnt += 1
            else:
                break
    print("%d images to process"%cnt)
    return file_path_list


def feature_extractor(file_path):
    '''
    Description: extract feature from given image
    Input: file_path - image path
    Output: des - a list of descriptors of all the keypoint from the image
    '''
    detector = cv2.ORB_create(nfeatures=1000)
    img = cv2.imread(file_path)
    kp = detector.detect(img, None) #find the keypoint
    _, des = detector.compute(img, kp) #compute the descriptor

    return des


def sift_extractor(file_path):
    '''
    Description: extract \emph{sift} feature from given image
    Input: file_path - image path
    Output: des - a list of descriptors of all the keypoint from the image
    '''
    img = cv2.imread(file_path)
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    _,des = sift.detectAndCompute(gray,None) 

    return des


def get_cluster_center(des_set, K):
    '''
    Description: cluter using a default setting
    Input: des_set - cluster data
                 K - the number of cluster center
    Output: laber  - a np array of the nearest center for each cluster data
            center - a np array of the K cluster center
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    des_set = np.float32(des_set)
    ret, label, center = cv2.kmeans(des_set, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    return label, center


def get_des_vector(file_path_list):
    '''
    Description: get descriptors of all the images 
    Input: file_path_list - all images path
    Output:       all_des - a np array of all descriptors
            iamge_des_len - a list of number of the keypoints for each image 
    '''
    all_des = np.empty(shape=[0, DESDIM])
    image_des_len = []

    for eachFile in file_path_list:
        try:
            #des = feature_extractor(eachFile)
            des = feature_extractor(eachFile)
            all_des = np.concatenate([all_des, des])
            image_des_len.append(len(des))
            print(eachFile)
        except:
            print(eachFile)
            #image_des_len.append(0)
            print("extract feature error")
    return all_des, image_des_len


def get_codebook(all_des, K):
    '''
    Description: train the codebook from all of the descriptors
    Input: all_des - training data for the codebook
                 K - the column of the codebook

    '''
    label, center = get_cluster_center(all_des, K)
    return label, center


def get_vlad_base(img_des_len, NNlabel, all_des, codebook):
    '''
    Description: get all images vlad vector 
    '''
    cursor = 0
    vlad_base = []
    for eachImage in img_des_len:
        descrips = all_des[cursor : cursor + eachImage]
        centriods_id = NNlabel[cursor : cursor + eachImage]
        centriods = codebook[centriods_id]
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
        cursor += eachImage
    
        vlad_norm = vlad.copy()
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
        vlad_base.append(vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1))

    return vlad_base


def get_vlad_base_pca(img_des_len, NNlabel, all_des, codebook):
    cursor = 0
    vlad_base = []
    for eachImage in img_des_len:
        descrips = all_des[cursor : cursor + eachImage]
        centriods_id = NNlabel[cursor : cursor + eachImage]
        centriods = codebook[centriods_id]
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
        cursor += eachImage

        vlad_base.append(vlad.reshape(COLUMNOFCODEBOOK * DESDIM, -1))
    
    vlad_base_pca = np.array(vlad_base)
    vlad_base_pca = vlad_base_pca.reshape(-1, DESDIM * COLUMNOFCODEBOOK)
    sklearn_pca = sklearnPCA(n_components=PCAD)
    sklearn_transf = sklearn_pca.fit_transform(vlad_base_pca)
    sklearn_transf_norm = sklearn_transf.copy()
    for each, each_norm in zip(sklearn_transf, sklearn_transf_norm):
        cv2.normalize(each, each_norm, 1.0, 0.0, cv2.NORM_L2)
    return sklearn_transf_norm, sklearn_pca

def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)


def get_pic_vlad(pic, des_size, codebook):
    '''
    Description: get the vlad vector of each image
    '''
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = 1000000000.0
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind]
    
    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
    vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)
    
    return vlad_norm

def get_pic_vlad_pca(pic, des_size, codebook, sklearn_pca):
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = 1000000000.0
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind]
    
    vlad = vlad.reshape(-1, COLUMNOFCODEBOOK * DESDIM)
    sklearn_transf = sklearn_pca.transform(vlad)
    sklearn_transf_norm = sklearn_transf.copy()
    cv2.normalize(sklearn_transf, sklearn_transf_norm, 1.0, 0.0, cv2.NORM_L2)
    
    return sklearn_transf_norm

def get_PQ_dist(vec1, vec2):
    vec1 = np.hsplit(vec1, SUBVEC)
    vec2 = np.hsplit(vec2, SUBVEC)
    dist = 0.0
    for sv1, sv2 in zip(vec1, vec2):
        dist += cal_vec_dist(sv1, sv2)
    return dist

def get_adc_dist(vec1, vec2):
    indexdist = float(sum(ret_adc_code[vec1] == label_list[vec2]))
    pqdist = get_PQ_dist(vlad_base_pca[vec2], ret_vlad_list[vec1])
    if indexdist <= 2:
        return pqdist / 3.0
    else:
        return pqdist / indexdist



##get all the descriptor vectors of the data set
file_path_list = read_file()
all_des, image_des_len = get_des_vector(file_path_list)

##get all the descriptor vectors of the query set
retrieval_image_path = read_file("D://datas//query//")
ret_des, ret_des_len = get_des_vector(retrieval_image_path)

##trainning the codebook
NNlabel, codebook = get_codebook(all_des, COLUMNOFCODEBOOK)


####if testtype == 0,vlad only
if TESTTYPE == 0: 

    vlad_base = get_vlad_base(image_des_len, NNlabel, all_des, codebook)

    ##get all the vlad vectors of retrival set without pca dimensionality reduction
    cursor_ret = 0
    ret_vlad_list = []
    for eachretpic in range(len(ret_des_len)):
        pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
        ret_vlad = get_pic_vlad(pic, ret_des_len[eachretpic], codebook)
        cursor_ret += ret_des_len[eachretpic]
        ret_vlad_list.append(ret_vlad)

    ##test and evaluation
    top1_cnt = 0
    for i in range(len(ret_vlad_list)):
        dist_list = []
        print("%dth image" % (i + 1000))
        for eachpic in range(len(image_des_len)):
            dist = cal_vec_dist(ret_vlad_list[i], vlad_base[eachpic])
            dist_list.append(dist)
        
        most_sim = np.array(dist_list)

        #choose the three nearest images of the given image 
        z = heapq.nsmallest(3,most_sim)
        index_first = dist_list.index(z[0])
        index_second = dist_list.index(z[1])
        index_third = dist_list.index(z[2])
       
        top1 = file_path_list[index_second][22:26]
        if top1 == str(i + 1000):
            top1_cnt += 1
        print("the %s is the first sim,the distance is the %f"%(file_path_list[index_second], z[1]))
        print("the %s is the second sim,the distance is the %f"%(file_path_list[index_third], z[2]))

    print(top1_cnt/500.0)  

####if testtype == 1,vlad + pca
elif TESTTYPE == 1:
    vlad_base_pca, sk_pca = get_vlad_base_pca(image_des_len, NNlabel, all_des, codebook)

    ##get all the vlad vectors of retrival set with pca dimensionality reduction
    cursor_ret = 0
    ret_vlad_list = []
    for eachretpic in range(len(ret_des_len)):
        pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
        ret_vlad = get_pic_vlad_pca(pic, ret_des_len[eachretpic], codebook, sk_pca)
        cursor_ret += ret_des_len[eachretpic]
        ret_vlad_list.append(ret_vlad)

    ### evaluate the performance of the retrival set
    top1_cnt = 0
    for i in range(len(ret_vlad_list)):
        dist_list = []
        print("%dth image" % (i + 1000))
        for eachpic in range(len(image_des_len)):
            dist = cal_vec_dist(vlad_base_pca[eachpic], ret_vlad_list[i])
            dist_list.append(dist)
        
        most_sim = np.array(dist_list)
        z = heapq.nsmallest(3,most_sim)
        index_first = dist_list.index(z[0])
        index_second = dist_list.index(z[1])
        index_third = dist_list.index(z[2])
        
        top1 = file_path_list[index_second][22:26]
        if top1 == str(i + 1000):
            top1_cnt += 1
        print("the %s is the second sim,the distance is the %f"%(file_path_list[index_second][22:28], z[1]))
        print("the %s is the third sim,the distance is the %f"%(file_path_list[index_third][22:28], z[2]))

    print(top1_cnt/500.0) 

else:

    vlad_base_pca, sk_pca = get_vlad_base_pca(image_des_len, NNlabel, all_des, codebook)

    ##get all the vlad vectors of retrival set with pca dimensionality reduction
    cursor_ret = 0
    ret_vlad_list = []
    for eachretpic in range(len(ret_des_len)):
        pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
        ret_vlad = get_pic_vlad_pca(pic, ret_des_len[eachretpic], codebook, sk_pca)
        cursor_ret += ret_des_len[eachretpic]
        ret_vlad_list.append(ret_vlad)

    ###using the adc method to retrive the img
    '''
    1. divide the vlad vector to subvector
    2. cluster each subvector
    3. using the index of the nearest cluster center to denote the subvector
    4. concat all index of subvector to denote the vlad vector
    '''
    sub_vlad_base = np.hsplit(vlad_base_pca, SUBVEC)
    label_list = []
    center_list = []
    for eachcluster in range(SUBVEC):
        label, center = get_cluster_center(sub_vlad_base[eachcluster], SUBCLUSTER)
        label_list.append(label)
        center_list.append(center)
    label_list = np.array(label_list)
    label_list = label_list.reshape(SUBVEC, -1)
    label_list = label_list.T

    ret_adc_code = []
    for eachretimg in range(len(ret_vlad_list)):
        pic_vlad = ret_vlad_list[eachretimg]
        sub_vec = np.hsplit(pic_vlad, SUBVEC)
        adc_code = []
        for eachSubVec, eachSubCluster in zip(sub_vec, center_list):
            min_dist = 1000000000.0
            ind = 0
            for i in range(SUBCLUSTER):
                dist = cal_vec_dist(eachSubVec, eachSubCluster[i])
                if dist < min_dist:
                    min_dist = dist
                    ind = i
            adc_code.append(ind)
        ret_adc_code.append(adc_code)

    ### evaluate the performance of the retrival set
    cnt = 0
    for i in range(len(ret_adc_code)):
        dist = []
        for j in range(len(label_list)):
            dis = get_adc_dist(i, j)
            dist.append(dis)

        most_sim = np.array(dist)
        
        zz = heapq.nsmallest(3,most_sim)
        index_first = dist.index(zz[0])
        index_second = dist.index(zz[1])
        index_third = dist.index(zz[2])
        
        print(i + 1000)
        print("%s is the most similar img\n"% file_path_list[index_second][22:26])
        
        if(file_path_list[index_second][22:26] == str(i + 1000)):
            cnt += 1
    print(cnt/500.0)