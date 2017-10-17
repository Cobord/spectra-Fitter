import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

spectraDF=0
GMMFitDF=0

samplePeakData=[[10,450,20],[10,600,20]]

def makeArtificialDF(numGridPoints=10,lowWavelength=400,highWavelength=700):
    # make an empty dataframe for a spectra that has numGridPoints
    #   from lowWavelength to highWavelength as potential data
    wavelengths = np.linspace(lowWavelength,highWavelength,numGridPoints)
    columns = np.append(wavelengths,'label')
    initializeData = np.zeros((0,numGridPoints+1))
    df = pandas.DataFrame(initializeData,columns=columns)
    global spectraDF
    spectraDF=df
    return df

def gauss(x, *p):
    # Define model function to be used to make artificial spectra
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def sumOfGaussians(x,ps):
    # sum of gaussians
    result=0
    for p in ps:
        result+=gauss(x,*p)
    return result

def sumOfThreeGaussians(x,p11,p12,p13,p21,p22,p23,p31,p32,p33):
    return sumOfGaussians(x,[[p11,p12,p13],[p21,p22,p23],[p31,p32,p33]])

def testSumOfGaussians():
# Test for sumOfGaussians
    allX=np.linspace(-10,10,100)
    ps=[[10,-3,1],[10,4,1],[10,7,1]]
    allY=list(map(lambda x: sumOfGaussians(x,ps),allX))
    y_noise = 0.5 * np.random.normal(size=len(allY))
    allY=allY+y_noise
    coeff, var_matrix = curve_fit(sumOfThreeGaussians, allX, allY,p0=sum(ps,[]))
    allY2=list(map(lambda x:sumOfThreeGaussians(x,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4],coeff[5],coeff[6],coeff[7],coeff[8]),allX))
    plt.plot(allX,allY)
    plt.plot(allX,allY2)
    return coeff

def getWaveLengthsUsed(df=spectraDF):
    # reading from the column names of spectraDF
    # see which wavelengths are used as data points
    waveLengthsUsed = []
    for col in df.columns.values:
        try:
            float(col)
            waveLengthsUsed.append(float(col))
        except ValueError:
            pass
    return waveLengthsUsed

def fillArtificialSpectra(peakData=samplePeakData,toLabel='none',df=spectraDF):
    # provide an (3,N) array peakData that stores ideal amplitudes,locations and widths
    #   use that to make a spectra that resembles that ideal
    #   and input that into the DataFrame with label toLabel
    waveLengthsUsed=getWaveLengthsUsed(df)
    simulatedSpectra=0*np.array(waveLengthsUsed)
    for p in peakData:
        toAdd=list(map(lambda x: gauss(x,*p),waveLengthsUsed))
        simulatedSpectra+=np.array(toAdd)
        #print(simulatedSpectra)
    #plt.plot(waveLengthsUsed,simulatedSpectra)
    newRow=np.append(simulatedSpectra,toLabel)
    df.loc[len(df)] = newRow
    global spectraDF
    spectraDF=df
    return df

def getColumnNames(numPeaks):
    # return an array of form peakAmp1,peakLoc1,peakWidth1,peakAmp2
    #   etc all the way to numPeaks
    peakAmp=lambda x: 'peakAmp'+str(x)
    peakLoc=lambda x: 'peakLoc'+str(x)
    peakWidth=lambda x: 'peakWidth'+str(x)
    columnNames=list(map(lambda x: [peakAmp(x),peakLoc(x),peakWidth(x)],np.arange(1,numPeaks+1)))
    columnNames=sum(columnNames,[])
    return np.append(columnNames,'label')

def makeEmptyFitDF(numPeaks):
    # makes an empty DF for the GMM fit data to go into
    # as well as the column for the label
    columnNames=getColumnNames(numPeaks)
    initializeData = np.zeros((0,3*numPeaks+1))
    df1 = pandas.DataFrame(initializeData,columns=columnNames)
    global GMMFitDF
    GMMFitDF=df1
    return df1

#def GaussMixModelEachRow
# return df1 with the GMM fit parameters of each row and the label
    

#def kMeans on df1
# ignore the label and do a kMeans cluster
# see if we get the same clustering as the original labels