# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:10:44 2019

@author: nico_
"""
import time
from imutils.video import VideoStream
import argparse
import face_recognition
import cv2
import numpy as np
import glob
import os
import logging
import pyrealsense2 as rs

import time

from gtts import gTTS
import pygame
import io

from mia_entranaminhacasa import pensa
from miaSTT_v3 import STT
def say(audio):
    tts = gTTS(audio, lang='pt-br')
    pygame.mixer.init()
    pygame.init()  # this is needed for pygame.event.* and needs to be called after mixer.init() otherwise no sound is played 
    with io.BytesIO() as f: # use a memory stream
        tts.write_to_fp(f)
        f.seek(0)
        pygame.mixer.music.load(f)
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        pygame.event.set_allowed(pygame.USEREVENT)
        pygame.mixer.music.play()
        pygame.event.wait() # play() is asynchronous. This wait forces the speaking to be finished before closing f and returning

def get_face_embeddings_from_image(image, convert_to_rgb=False):
    """
    Take a raw image and run both the face detection and face embedding model on it
    """
    # Convert from BGR to RGB if needed
    if convert_to_rgb:
        image = image[:, :, ::-1]

    # run the face detection model to find face locations
    face_locations = face_recognition.face_locations(image)

    # run the embedding model to get face embeddings for the supplied locations
    face_encodings = face_recognition.face_encodings(image, face_locations)

    return face_locations, face_encodings

def setup_database():
    """
    Load reference images and create a database of their face encodings
    """
    database = {}

    for filename in glob.glob(os.path.join('./images', '*.jpg')):
        # load image
        image_rgb = face_recognition.load_image_file(filename)

        # use the name in the filename as the identity key
        #[0] desde da imagem 0
        nomeprinc = os.path.splitext(os.path.basename(filename))[0]
        identity = nomeprinc[5:]


        # get the face encoding and link it to the identity
        locations, encodings = get_face_embeddings_from_image(image_rgb)
        if len(encodings) > 0:
            database[identity] = encodings[0]
        else:
            print("Procurando...")
            #quit()
            
    return database


def atualiza_database(name, database):
    """
    Load reference images and create a database of their face encodings
    """

    database_atual = database

    for filename in glob.glob(os.path.join('./images', '*.jpg')):
        testname = filename[14:]
        testname = testname[:-4]
        if testname == name:
            # load image
            image_rgb = face_recognition.load_image_file(filename)

            # use the name in the filename as the identity key
            #[0] desde da imagem 0
            nomeprinc = os.path.splitext(os.path.basename(filename))[0]
            identity = nomeprinc[5:]

            # get the face encoding and link it to the identity
            locations, encodings = get_face_embeddings_from_image(image_rgb)
            if len(encodings) > 0:
                database_atual[identity] = encodings[0]
            else:
                print("Procurando...")
                #quit()
    return database_atual

def tem_face(frame, location, name=None):
    """
    Paint a rectangle around the face and write the name
    """
    # unpack the coordinates from the location tuple
    top, right, bottom, left = location
    cara=1
    if name is None:
        name = 'Unknown'
        color = (0, 0, 255)  # red for unrecognized face
    else:
        color = (0, 128, 0)  # dark green for recognized face

    # Draw a box around the face
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Draw a label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
    return cara

def Captura_frames():
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    frame = np.asanyarray(color_frame.get_data())

    orig = frame.copy()

    return frame, orig

def TiraFoto(orig,nome):
    totalFotos = 0
    total = 0
    for foto in glob.glob(os.path.join('./images', '*.jpg')):
        total += 1
    
    p = '.\images'+str(total).zfill(5)+nome+'.jpg'####################
    start_time = time.time()
    cv2.imwrite(p, orig)
    print("fotos capturadas: ", totalFotos)
    print("total no BD: ", total)

    while totalFotos <= 10:
        if (time.time() - start_time)>0.5:
            p = './images/'+str(total).zfill(5)+nome+'.jpg'###############
            orig, frame = Captura_frames()

            cv2.imwrite(p, orig)
            total += 1
            totalFotos = totalFotos + 1
            print("fotos capturadas: ", totalFotos)
            start_time = time.time()

    print("[INFO] {} face images stored".format(total))
    return total
        
def fdata():
    database = setup_database()
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())
    return database, known_face_encodings, known_face_names

def fadata(name, database):
    database_atual = atualiza_database(name, database)
    known_face_encodings = list(database.values())
    known_face_names = list(database.keys())

    return database_atual, known_face_encodings, known_face_names

def frs():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    return pipeline

com_foto=0
loopnome = 1
continua = 1
contador = 0

pipeline = frs()

a=0

proj_id = 'newagent-jekqnf'
session_id =  "123456789" 

incremento = 0 

numUnknow = 0
name = 'usuário'
while continua ==1:
    #print(database)
    
    face=0
    loopnome = 1
    
    #print('incremento:',incremento)
    #print('numU:',numUnknow)
    '''
    if incremento >= 20:
        numUnknow = 0
        incremento = 0
    '''
    pergunta = STT()
    if pergunta != None:
        imagem = ''
        estado = 2
        with open('pipe.txt','w') as f:
            now_ = str(estado) + ';' + pergunta + ';' + imagem 
            f.write('')
            f.write(now_)
        resposta, estado, imagem = pensa(pergunta,name)
        time.sleep(1)
        #estado = 1
        #imagem = '.\Imagens_Paes e Mapas\Achados e Perdidos.png'
        with open('pipe.txt','w') as f:
            now_ = str(estado) + ';' + resposta + ';' + imagem 
            f.write('')
            f.write(now_)
        
        print(resposta)
        say(resposta)
        
        if resposta == 'Ta bom, qual seu nome?':
            print('batata')
            orig, frame = Captura_frames()
            face_locations, face_encodings = get_face_embeddings_from_image(frame, convert_to_rgb=False)
            for location, face_encoding in zip(face_locations, face_encodings):
                distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if np.any(distances <= 0.4):
                    best_match_idx = np.argmin(distances)
                    name = known_face_names[best_match_idx]
                else:
                    name = None
        
                face= tem_face(frame, location, name)
            prazer = 'Prazer '+str(nome)+'!'
            estado=2
            imagem = ''
            with open('pipe.txt','w') as f:
                now_ = str(estado) + ';' + prazer + ';' + imagem 
                f.write('')
                f.write(now_)
            say(prazer)
            with open('pipe.txt','w') as f:
                now_ = str(estado) + ';Um segundo, estou te registrando.;' + imagem 
                f.write('')
                f.write(now_)
            say('Um segundo, estou te registrando')
            print('Um segundo, estou te registrando')
            total = TiraFoto(orig,nome)
            database_atual, known_face_encodings, known_face_names = fadata(nome,database)
            with open('pipe.txt','w') as f:
                now_ = str(estado) + ';Registrado.;' + imagem 
                f.write('')
                f.write(now_)
            say('Registrado')
        
        '''  
        else:
            numUnknow = numUnknow+1
            if numUnknow >10:
                numUnknow = 0
                incremento = 0
                while loopnome == 1:
                    estado = 2
                    imagem = ''
                    with open('pipe.txt','w') as f:
                            now_ = str(estado) + ';Qual o seu nome?;' + imagem 
                            f.write('')
                            f.write(now_)
                    say('Qual o seu nome?')
                    #pergunta=input('Qual o seu nome?')
                    pergunta = STT()
                    if pergunta == ('Sair' or 'sair'):
                        loopnome = 0
                        break
                    if pergunta != None:
                        #aparecer texto do usuario
                        with open('pipe.txt','w') as f:
                            now_ = str(estado) +';'+ pergunta + ';' + imagem 
                            f.write('')
                            f.write(now_)
                        time.sleep(1)
                        txt1 = pergunta.split()
                        tamanho = len(txt1)
                        if tamanho == 1:
                            nome = txt1[0]
                        else:
                            resposta, estado, imagem = pensa(pergunta,name)
                            corte = len(resposta)-1
                            nome = resposta[7:corte]
                            #print(nome)
                            
                        conf = 'Seu nome é '+ str(nome)+'?'
                        estado=2
                        imagem = ''
                        with open('pipe.txt','w') as f:
                            now_ = str(estado) + ';' + conf + ';' + imagem 
                            f.write('')
                            f.write(now_)
                        say(conf)
                        print('Seu nome é',nome,'?')
                        
                        resposta = STT()
                        
                        #resposta = input()
                        #print(resposta)
                        if resposta != None:
                            estado = 2
                            imagem = ''
                            with open('pipe.txt','w') as f:
                                now_ = str(estado) +';'+ resposta + ';' + imagem 
                                f.write('')
                                f.write(now_)
                            time.sleep(1)
                            if resposta == ('sim' or 'Sim'):
                                #### se der tempo fazer pensa para essa resposta ou adicionar mais coisas no 'or'
                                prazer = 'Prazer '+str(nome)+'!'
                                estado=2
                                imagem = ''
                                with open('pipe.txt','w') as f:
                                    now_ = str(estado) + ';' + prazer + ';' + imagem 
                                    f.write('')
                                    f.write(now_)
                                say(prazer)
                                with open('pipe.txt','w') as f:
                                    now_ = str(estado) + ';Um segundo, estou te registrando.;' + imagem 
                                    f.write('')
                                    f.write(now_)
                                say('Um segundo, estou te registrando')
                                print('Um segundo, estou te registrando')
                                total = TiraFoto(orig,nome)
                                database_atual, known_face_encodings, known_face_names = fadata(nome,database)
                                with open('pipe.txt','w') as f:
                                    now_ = str(estado) + ';Registrado.;' + imagem 
                                    f.write('')
                                    f.write(now_)
                                say('Registrado')
                                
                                
                                loopnome=0
                    else:
                        with open('pipe.txt','w') as f:
                            now_ = str(estado) + ';Desculpe, não te escutei;' + imagem 
                            f.write('')
                            f.write(now_)
                        say('Desculpe, não te escutei')
            '''
    key = cv2.waitKey(1) & 0xFF#############
    if key == ord("q"):#############
        break################
    
    incremento = incremento+1
                
cv2.destroyAllWindows()
pipeline.stop()

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    