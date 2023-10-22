from main import listen
import speech_recognition as sr
import os
#import serial

#ser = serial.Serial("COM6", 9600, timeout = 1)

#def sendData(uInput):
#	ser.write(bytes(uInput, 'ascii'))

list2check = ["yukarı bak", "aşağı bak", "sola dön", "sağa dön", "ortaya bak", "hızlan", "yavaşla", "üst falso", "alt falso", "tempoyu arttır", "tempoyu düşür", "başla", "durdur", "dur", "mod 1", "mod 2", "mod 3","mod 4", "oyun modu"]
st_flag = False
r = sr.Recognizer()
with sr.Microphone(device_index=1) as source:
    r.adjust_for_ambient_noise(source)

with open('./comm/comm.txt', 'w') as f:
    f.writelines(["idle"])

while True:
    phrase = listen(r)
    if phrase in list2check:
        if phrase == "başla":
            print("S")
            st_flag = True
            with open('./comm/comm.txt', 'w') as f:
                f.writelines(["start"])

        if st_flag:
            if phrase == "yukarı bak":
                print("U")
            if phrase == "aşağı bak":
                print("D")
            if phrase == "sola dön":
                print("L")
            if phrase == "sağa dön":
                print("R")
            if phrase == "ortaya bak":
                print("M")
            if phrase == "hızlan":
                print("F")
            if phrase == "yavaşla":
                print("Y")
            if phrase == "üst falso":
                print("Q")
            if phrase == "alt falso":
                print("W")
            if phrase == "tempoyu arttır":
                print("T")
            if phrase == "tempoyu düşür":
                print("N")
            if phrase == "mod 1":
                print("1")
            if phrase == "mod 2":
                print("2")
            if phrase == "mod 3":
                print("3")
            if phrase == "mod 4":
                print("4")
            if phrase == "oyun modu":
                print("G")
            if phrase == "durdur" or phrase == "dur":
                print("X")
                with open('./comm/comm.txt', 'w') as f:
                    f.writelines(["stop"])
                st_flag = False



