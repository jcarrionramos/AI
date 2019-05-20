import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print('Di algo!')
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="es-CL")
        print(text)
    except:
        print('error')
