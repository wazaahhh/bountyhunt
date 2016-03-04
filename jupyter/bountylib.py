import numpy as np
import pylab as pl
import pandas as pd

import os,sys
from dateutil.parser import parse
from datetime import datetime
import time
from collections import Counter


sys.path.append("/Users/maithoma/work/python/")
from tm_python_lib import *
from fitting_tools import *
from adaptive_kernel_tom import adaptive_kernel_quantile_pdf_tom

fig_width_pt = 420.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0 / 72.27  # Convert pt to inch
golden_mean = (np.sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
fig_width = fig_width_pt * inches_per_pt  # width in inches
fig_height = fig_width  # *golden_mean      # height in inches
fig_size = [fig_width, fig_height]


params = {'backend': 'ps',
          'axes.labelsize': 25,
          'text.fontsize': 32,
          'legend.fontsize': 14,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
          'figure.figsize': fig_size}
pl.rcParams.update(params)

rootdir = "/Users/maithoma/work/github/bountyhunt/jupyter/"
datadir = rootdir + "data/"
figuredir = rootdir + "figures/"


def loaddata():
    '''load data from csv file into pandas dataframe'''
    df = pd.DataFrame.from_csv(datadir + "all.csv")
    df['datetime'] = np.array([datetime.strptime(dt,"%m/%d/%Y %H:%M:%S") for dt in df['Timestamp']])
    df['timestamp'] = np.array([time.mktime(datetime.strptime(dt,"%m/%d/%Y %H:%M:%S").timetuple()) for dt in df['Timestamp']])
    df.drop('Timestamp', axis=1, inplace=True)
    df.index = df.datetime
    df = df.sort_index()
    return df



'''Plots'''

def plotPowerLawFit(loss,xmin=1,continuousFit=True,addnoise=False,confint=.01,plot=False):
    '''General power law plotting method from
    continuous data or discrete data with noise added
    '''


    loss,rank = rankorder(loss)
    y = rank

    if addnoise:
        x = loss + np.random.rand(len(loss)) - 0.5
    else:
        x = loss



    '''Normalized plot of the empirical distribution'''
    rankNorm = rank/float(rank[-1])
    rankMin = rankNorm[loss <= loss][-1]


    '''Plot of the fitted distribution'''
    mu,confidence,nPoints = pwlaw_fit_cont(x,xmin)
    print mu,confidence,nPoints

    xFit = np.logspace(np.log10(xmin),np.log10(max(loss)))
    yFit = (rankMin-0.03)*1/(xFit/float(xmin))**mu

    if plot:
        pl.loglog(loss,rankNorm,'k.' ,alpha=0.5)
        pl.loglog(xFit,yFit,'k-.')


    '''Add confidence intervals'''

    if confint:
        m,L,U,pvalue = bootstrapping(x,xmin,confint=confint,numiter = -1)
        x,y = rankorder(L)
        yLowerNorm = y/float(y[-1])
        #pl.loglog(x,yMin*yLowerNorm,'m')
        x,y = rankorder(U)
        yUpperNorm = y/float(y[-1])


    return {'x':loss,'y':rankNorm,'xFit':xFit,'yFit':yFit}

def bountyPerResearcher(df):
    '''Distribution of Bounties awarded per researcher
    and distribution of bounties awarded per program per researcher.
    '''

    pl.figure(1,(16,7))

    pl.subplot(121)
    '''Bounty Awards per Program'''
    loss = df.Bounty.groupby([df.Program]).sum().values
    dic = plotPowerLawFit(loss,xmin=3000,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],100,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'go',label="Sum awards per Program")
    #pl.loglog(dic['x'],dic['y'],'.',color='blue',alpha=0.5)
    pl.loglog(dic['xFit'],dic['yFit']*0.1,'k--',lw=2)

    '''Bounty Awards per Researcher'''
    loss = df.Bounty.groupby([df.Researcher]).sum().values
    dic = plotPowerLawFit(loss,xmin=3000,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],100,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'bo',label="Sum awards per Researcher")
    #pl.loglog(dic['x'],dic['y'],'.',color='blue',alpha=0.5)
    pl.loglog(dic['xFit'],dic['yFit']*0.1,'k--',lw=2)

    '''Bounty Awards per Researcher per Program'''
    loss = df.Bounty.groupby([df.Program,df.Researcher]).sum().values
    dic = plotPowerLawFit(loss,xmin=3000,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],100,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'ro',label="Sum awards per Researcher per Researcher")
    #pl.loglog(dic['x'],dic['y'],'.',color='red',alpha=0.5)
    pl.loglog(dic['xFit'],dic['yFit']*0.05,'k--',lw=2)

    pl.xlabel("Sum Bounty Awards")
    pl.ylabel("CCDF")

    pl.legend(loc=0)
    #pl.ylim(ymin=5*10**-4)
    #pl.xlim(xmax=3*10**5)

    pl.subplot(122)
    '''Bounty Count per Program'''
    loss = df.Bounty.groupby([df.Program]).count().values
    dic = plotPowerLawFit(loss,xmin=20,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],40,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'go',label="Bounties per Program")
    #pl.loglog(dic['x'],dic['y'],'.',color=color,alpha=0.5,label="Bounties per Researcher")
    pl.loglog(dic['xFit'],dic['yFit']*0.7,'k--')#,color='blue',label="Fit Bounties per Researcher")


    '''Bounty Count per Researcher'''
    loss = df.Bounty.groupby([df.Researcher]).count().values
    dic = plotPowerLawFit(loss,xmin=3,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],40,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'bo',label="Bounties per Researcher")
    #pl.loglog(dic['x'],dic['y'],'.',color=color,alpha=0.5,label="Bounties per Researcher")
    pl.loglog(dic['xFit'],dic['yFit']*0.7,'k--')#,color='blue',label="Fit Bounties per Researcher")

    '''Bounty Count per Researcher per Program'''
    loss = df.Bounty.groupby([df.Program,df.Researcher]).count().values
    dic = plotPowerLawFit(loss,xmin=1,addnoise=True,confint=0)
    B = binning(dic['x'],dic['y'],40,log_10=True,confinter=5)
    pl.loglog(10**B['bins'],10**B['mean'],'ro',label="Bounties per Researcher per Program")
    #pl.loglog(dic['x'],dic['y'],'.',color='red',alpha=0.5,label="Bounties per program per Researcher")
    pl.loglog(dic['xFit'],dic['yFit']*0.7,'k--',lw=2)#,color='red',label="Fit Bounties per program per Researcher")

    pl.legend(loc=0)
    pl.xlabel("Bounty Count")
    pl.ylabel("CCDF")

    #pl.xlim(xmax=60)
    #pl.ylim(ymin=5*10**-4)




    pl.savefig(figuredir + "CCDF_count_Bounties.eps")


def plotFit(loss,xmin=1,label=""):

    pl.close("all")
    pl.figure(1,(25,9))

    index,mu,error,llh,points = vary_threshold(loss,xmin=0.05,index=100,type='cont')

    pl.subplot(131)
    pl.semilogx(index,mu)
    pl.semilogx(index,mu + error)
    pl.semilogx(index,mu - error)
    pl.xlabel("threshold")
    pl.ylabel("exponent mu")

    pl.subplot(132)
    pl.semilogx(index,llh)
    pl.xlabel("threshold")
    pl.ylabel("Loglihelihood")

    pl.subplot(133)
    #xmin = index[np.argmax(llh)]
    mu,confidence,n = pwlaw_fit_cont(loss,xmin)

    print "nEvents: %.2f, %.2f , %.2f"%(len(loss),len(loss[loss<xmin]),len(loss[loss>=xmin]))
    print "sumLoss: %.2f, %.2f , %.2f"%(sum(loss),sum(loss[loss<xmin]),sum(loss[loss>=xmin]))
    print "min max: %.2f, %.2f"%(min(loss),max(loss))
    whatever = bootstrapping(loss,xmin,confint=.01,plot=True)
    print "fit: %s (p-value = %s, std.err = %.2f)"%(mu,whatever[-1],confidence)

    pl.xlabel("Loss")
    pl.ylabel("Rank Ordering")
    pl.title(label)

def plotTimeline(df):
    t_resol = "1W"
    ax = pl.figure(1,(20,5))
    color = ['yellow','cyan','green','lime','red','magenta','purple','blue','grey']

    x_old = df.Bounty.resample(t_resol,how="count").index
    y_old = np.zeros_like(df.Bounty.resample(t_resol,how="count").values)
    y_others = np.zeros_like(df.Bounty.resample(t_resol,how="count").values)

    i=0
    for program in df.Program.unique():
        dfprog = df[df.Program == program]
        countBountiesProg = dfprog.Bounty.resample(t_resol,how="count")
        X = countBountiesProg.index
        i0 = np.argwhere(X[0] == x_old)[0]
        Y = countBountiesProg.values

        if df.Program[df.Program == program].count() < 90:
            y_others[i0:i0+len(X)] += Y
            continue

        pl.bar(X[0],80,width=7,color=color[i],lw=0.0,alpha=0.05)
        iMax = np.argmax(Y)
        #pl.bar(countBountiesProg.index[iMax],80,width=7,color=color[i],lw=0.0,alpha=0.2)
        pl.bar(X,Y,width=7,bottom=y_old[i0:i0+len(X)],lw=0.05,color=color[i],label=program)
        y_old[i0:i0+len(X)] += Y
        i+=1

        #pl.bar(x_old,y_others,width=7,bottom=y_old,lw=0.05,color='lightgrey',label=program)

    pl.xlabel("Time [weeks]")
    pl.ylabel("(Cumulative) bounties awarded")
    pl.legend(loc=0)
    pl.savefig(figuredir + "timeline.eps")


def plotDecay(df):
    t_resol = "1W"

    pl.figure(2,(12,7))

    Xall = []
    Yall = []

    Xmax = []
    Xinit = []
    Xdiff = []

    for program in df.Program.unique():
        #print program
        dfprog = df[df.Program == program]
        countBountiesProg = dfprog.Bounty.resample(t_resol,how="count")
        Y = countBountiesProg.values
        i0 = np.argmax(Y)
        Xmax = np.append(Xmax,countBountiesProg.index[i0])
        Xinit = np.append(Xinit,countBountiesProg.index[0])
        Xdiff = np.append(Xdiff, (countBountiesProg.index[i0] - countBountiesProg.index[0]).days)
        X = (countBountiesProg.index - countBountiesProg.index[0]).days + 1
        #Y = Y/float(max(Y))
        Y = Y/float(Y[0])


        Xall = np.append(Xall,X)
        Yall = np.append(Yall,Y)

        c = (X > 0)*(Y > 0)
        lX = np.log10(X[c])
        lY = np.log10(Y[c])

        #pl.plot(lX,lY,label=program)

    Xall = Xall/7.
    c = (Xall > 0)*(Yall > 0)*(Xall < 3000)
    lXall = np.log10(Xall[c])
    lYall = np.log10(Yall[c])

    B = binning(lXall,lYall,50,confinter=10)
    #oB = np.argmax(B['median'])
    #print B['bins']
    #print B['median']
    fit = S.linregress(B['bins'],B['median'])
    print fit
    pl.plot(B['bins'],B['median'],'ro')
    #pl.plot(B['bins'],B['percUp'],'r--')
    #pl.plot(B['bins'],B['percDown'],'r--')
    pl.plot(B['bins'],B['bins']*fit[0] + fit[1],'k--')

    pl.xlabel("log10(Time) [weeks]")
    pl.ylabel("log10(Probability Density Function)")

    pl.savefig(figuredir + "decay.eps")


def bootstrapping(data,xmin,confint=.05,numiter = -1,plot=False,plotconfint=False):
    '''Bootstrapping power law distribution'''
    data = np.array(data) # make sure the input is an array
    sample = data[data >= xmin]
    mu,confidence,nPoints = pwlaw_fit_cont(sample,xmin) #fit original power law

    f = 1/(sample/float(xmin))**mu
    ksInit = kstest(sample,f)
    #print ksInit

    if nPoints==0:
        print "no value larger than %s"%xmin
        return

    if numiter == -1:
        numiter = round(1./4*(confint)**-2)

    m = np.zeros([numiter,nPoints])
    i = 0
    k = 0
    while i < numiter:
        q2 = pwlaw(len(sample),xmin,mu)[0]
        m[i]=np.sort(q2)
        ks = kstest(q2,f)

        if ks > ksInit:
            k += 1

        i+=1

    pvalue = k/float(numiter)
    U=np.percentile(m,100-confint*100,0)
    L=np.percentile(m,confint,0)

    if plot:
        x,y = rankorder(data)
        yNorm = y/float(y[-1])
        yMin = yNorm[x <= xmin][0]

        pl.loglog(x,yNorm,'k.')

        xFit = np.logspace(np.log10(xmin),np.log10(max(sample)))
        yFit = yMin*1/(xFit/float(xmin))**mu

        pl.loglog(xFit,yFit,'r-')

        if plotconfint:
            x,y = rankorder(L)
            yLowerNorm = y/float(y[-1])
            pl.loglog(x,yMin*yLowerNorm,'m')
            x,y = rankorder(U)
            yUpperNorm = y/float(y[-1])
            pl.loglog(x,yMin*yUpperNorm,'b')



    return m,L,U,pvalue

def kstest(sample1,sample2):
    return np.max(np.abs(sample1 - sample2))


def crossLagCorr(x,y,lagspan=35):

    rho = []
    L = range(-lagspan,lagspan)

    for l in L:
        if l==0:
            rho.append(S.spearmanr(x,y)[0])
        elif l < 0:
             rho.append(S.spearmanr(x[-l:],y[:l])[0])
        else:
            rho.append(S.spearmanr(x[:-l],y[l:])[0])

    return L,rho



def binning(x,y,bins,log_10=False,confinter=5):
    '''makes a simple binning'''

    x = np.array(x);y = np.array(y)

    if isinstance(bins,int) or isinstance(bins,float):
        bins = np.linspace(np.min(x)*0.9,np.max(x)*1.1,bins)
    else:
        bins = np.array(bins)

    if log_10:
        bins = bins[bins>0]
        c = x > 0
        x = x[c]
        y = y[c]
        bins = np.log10(bins)
        x = np.log10(x)
        y = np.log10(y)

    Tbins = []
    Median = []
    Mean = []
    Sigma =[]
    Perc_Up = []
    Perc_Down = []
    Points=[]


    for i,ix in enumerate(bins):
        if i+2>len(bins):
            break

        c1 = x >= ix
        c2 = x < bins[i+1]
        c=c1*c2

        if len(y[c])>0:
            Tbins = np.append(Tbins,np.median(x[c]))
            Median =  np.append(Median,np.median(y[c]))
            Mean = np.append(Mean,np.mean(y[c]))
            Sigma = np.append(Sigma,np.std(y[c]))
            Perc_Down = np.append(Perc_Down,np.percentile(y[c],confinter))
            Perc_Up = np.append(Perc_Up,np.percentile(y[c],100 - confinter))
            Points = np.append(Points,len(y[c]))


    return {'bins' : Tbins,
            'median' : Median,
            'mean' : Mean,
            'stdDev' : Sigma,
            'percDown' :Perc_Down,
            'percUp' :Perc_Up,
            'nPoints' : Points}
