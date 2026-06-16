#!/usr/bin/env python3
"""
Entrenamiento One-Class SVM con OpenCV.
Aprende solo de features autenticas. En inferencia, muestras lejanas del
hiperplano son FALSAS.

Uso:
    python train_oneclass.py --templates ./templates/ --autenticos ./training_data/autenticos/ --output ocsvm.xml
"""

import os, argparse
import numpy as np
import cv2
from pathlib import Path

TEMPLATE_SIZE = (100,100)
HIST_BINS=32; LAPLACIAN_BINS=32

ORB_FEATURES=2000; FLANN_CHECKS=100; RANSAC_THRESHOLD=5.0; MATCH_RATIO=0.75

class MultiTemplateAligner:
    def __init__(self, d):
        self.t={}; self.tk={}; self.td={}
        self.orb=cv2.ORB_create(nfeatures=ORB_FEATURES,scaleFactor=1.2,nlevels=1,edgeThreshold=3)
        ip=dict(algorithm=6,table_number=12,key_size=20,multi_probe_level=2)
        self.flann=cv2.FlannBasedMatcher(ip,dict(checks=FLANN_CHECKS))
        for f in sorted(Path(d).glob("*.png")):
            ft=f.stem; img=cv2.imread(str(f),cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img=cv2.resize(img,TEMPLATE_SIZE); self.t[ft]=img
            kp,des=self.orb.detectAndCompute(img,None)
            self.tk[ft]=kp if kp else []; self.td[ft]=des
    def align(self,img,ft):
        if ft not in self.t or img is None or img.size==0: return None
        ikp,ides=self.orb.detectAndCompute(img,None)
        if ides is not None and len(ides)>=4:
            try:
                td=self.td.get(ft)
                if td is not None and len(td)>=4:
                    m=self.flann.knnMatch(td,ides,k=2)
                    g=[a for a,b in m if a.distance<MATCH_RATIO*b.distance]
                    if len(g)>=4:
                        sp=np.float32([self.tk[ft][a.queryIdx].pt for a in g]).reshape(-1,1,2)
                        dp=np.float32([ikp[a.trainIdx].pt for a in g]).reshape(-1,1,2)
                        H,_=cv2.findHomography(dp,sp,cv2.RANSAC,RANSAC_THRESHOLD)
                        if H is not None:
                            return cv2.warpPerspective(img,H,TEMPLATE_SIZE,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
            except: pass
        return cv2.resize(img,TEMPLATE_SIZE)

def extract_features(img):
    h=cv2.calcHist([img],[0],None,[HIST_BINS],[0,256]); h=cv2.normalize(h,h).flatten()
    l=cv2.Laplacian(img,cv2.CV_64F); l=np.abs(l).astype(np.uint8)
    lh=cv2.calcHist([l],[0],None,[LAPLACIAN_BINS],[0,256]); lh=cv2.normalize(lh,lh).flatten()
    return np.concatenate([h,lh]).astype(np.float32)

def _detect_ft(path):
    known=["valor_","animal_","ir_","personaje_","serie_a"]
    for p in known:
        if path.stem.startswith(p):
            e=path.stem.find('_',len(p)); return path.stem[:e] if e!=-1 else path.stem[:len(p)]
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--templates",required=True)
    p.add_argument("--autenticos",required=True)
    p.add_argument("--output",default="ocsvm.xml")
    a=p.parse_args()

    aligner=MultiTemplateAligner(a.templates)
    X=[]
    for img_path in sorted(Path(a.autenticos).rglob("*.png")):
        ft=_detect_ft(img_path)
        if ft is None or ft=="serie_a": continue
        img=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        aligned=aligner.align(img,ft)
        if aligned is None: continue
        X.append(extract_features(aligned))
        if len(X)%100==0: print(f"  {len(X)}")

    X=np.array(X,dtype=np.float32)
    print(f"[DATA] {X.shape[0]} muestras, {X.shape[1]} dims")

    # One-Class SVM
    svm=cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_ONE_CLASS)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setGamma(1.0/X.shape[1])  # auto gamma
    svm.setNu(0.05)               # 5% outliers
    svm.train(X, cv2.ml.ROW_SAMPLE, np.ones(len(X), dtype=np.int32))

    # Evaluacion rapida: cuantas autenticas pasan
    y_pred=svm.predict(X)[1].flatten()
    passed=np.sum(y_pred==1)
    print(f"[EVAL] {passed}/{len(X)} autenticas pasan ({100*passed/len(X):.1f}%)")

    svm.save(a.output)
    print(f"[MODELO] {a.output}")

if __name__=="__main__":
    main()
