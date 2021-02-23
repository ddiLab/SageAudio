import pyaudio
import numpy as np
import wave
import argparse
import matplotlib.pyplot as plt

'''
usbaudio.py // Emily Brown // Feb 2021
Records audio from a USB microphone and saves in a .wav file. Optionally plots
wave and/or spectrum of input audio signal. To skip recording and analyze an input
file, use -i to specify input file and use in conjunction with -w and/or -s.

Options:
-o / --outpath  : specifies output file name - MUST end in .wav (default "out.wav")
-i / --inpath   : specifies input file name, skips recording - MUST end in .wav
-d / --duration : specifies recording length in seconds (default 5 seconds)
-w / --wave     : plot waveform of audio input
-z / --zoom     : plot waveform of first 0.05 seconds of audio input
-s / --spec     : plot spectrum of audio input

Based off guides by Joshua Hrisko:
https://makersportal.com/blog/2018/8/23/recording-audio-on-the-raspberry-pi-with-python-and-a-usb-microphone
https://makersportal.com/blog/2018/9/13/audio-processing-in-python-part-i-sampling-and-the-fast-fourier-transform
'''

# establish parser
parser = argparse.ArgumentParser(description="records audio from usb microphone and outputs in a .wav file")
parser.add_argument("-o", "--outpath", help="path to save output file to")
parser.add_argument("-i", "--inpath", help="skip recording and input .wav file")
parser.add_argument("-d", "--duration", help="duration in seconds to record", type=int)
parser.add_argument("-w", "--wave", help="plot audio waveform", action="store_true")
parser.add_argument("-z", "--zoom", help="plot audio waveform from 0 to 0.05 seconds", action="store_true")
parser.add_argument("-s", "--spec", help="plot audio spectrum", action="store_true")
args = parser.parse_args()

if args.outpath:
    wav_output_filename = args.outpath
else:
    wav_output_filename = 'out.wav'
    
if args.duration:
    record_secs = args.duration
else:
    record_secs = 5

if args.wave and args.zoom:
    print("error: --wave and --zoom mutually exclusive")
    exit()

#establish format parameters
form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
samp_period = 1/samp_rate
sig_len = samp_rate * record_secs
chunk = 4096 # 2^12 samples for buffer

# record unless input is overwridden
if args.inpath:
    wav_output_filename = args.inpath
else:
    # establish parameters for recording
    dev_index = 0
    audio = pyaudio.PyAudio()

    # create pyaudio stream
    stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                        input_device_index = dev_index,input = True, \
                        frames_per_buffer=chunk)
    print("recording for " + str(record_secs) + " seconds")
    frames = []

    # loop through stream and append audio chunks to frame array
    for ii in range(0,int((samp_rate/chunk)*record_secs)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording, output saved to " + wav_output_filename)

    # stop the stream, close it, and terminate the pyaudio instantiation
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # save the audio frames as .wav file
    wavefile = wave.open(wav_output_filename,'wb')
    wavefile.setnchannels(chans)
    wavefile.setsampwidth(audio.get_sample_size(form_1))
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

# analyze and plot signal
if args.wave or args.spec or args.zoom:
    # read wave file
    wavefile = wave.open(wav_output_filename)
    signal = wavefile.readframes(-1)
    signal = np.frombuffer(signal, np.int16)
    fs = wavefile.getframerate()

    plt.figure(1)
    wavefile.close()

    if args.wave or args.zoom:
        # create time vector for plotting in seconds
        time = np.linspace(0, len(signal) / samp_rate, num=len(signal))

        # normalize signal
        high, low = abs(max(signal)), abs(min(signal))
        norm_sig = signal / max(high, low)

        if args.spec: # build subplot
            plot_wave = plt.subplot(211)
            plot_wave.plot(time, norm_sig)
            if args.zoom:
                plot_wave.xlim([0,0.05])
        else : # build plot
            plt.title("Wave form (Time domain)")
            plt.plot(time, norm_sig)
            plt.xlabel('Time (s)')
            if args.zoom: # zoom in x axis
                plt.xlim([0,0.05])
    
    if args.spec:
        # FFT
        y, N = signal, len(signal)
        Y_k = np.fft.fft(y)[0:int(N/2)]/N # FFT function from numpy
        Y_k[1:] = 2*Y_k[1:] # take single-side spectrum
        Pxx = np.abs(Y_k) # erase imaginary part
        f = fs*np.arange((N/2))/N; # frequency vector

        if args.wave:  # build subplot
            plot_spec = plt.subplot(212)
            plot_spec.plot(f,Pxx)
        else: # build plot
            plt.title("Spectrum (Frequency domain)")
            plt.plot(f,Pxx,linewidth=5)
            plt.ylabel('Amplitude')
            plt.xlabel('Frequency (Hz)')
            plt.xlim([0, 22000])

    plt.show()

