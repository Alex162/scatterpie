import numpy as np # 1.22.3
import matplotlib #3.7.1
import matplotlib.patches as mpatches
import matplotlib.markers as mmarkers
import matplotlib.pyplot as plt
import numba as nb #0.56.4 (optional; speeds up setup_scatterpie() just comment out nb.njit() if you don't want it)
import time as time #only used for timing functions; simply comment out time.time calls etc.


#The above versions are what I know works; other versions may work, but no promises

"""
The default colourset (facecolourlist) is the colourlist of Paul Tol's 'light' colourscheme;
Highly reccomend checking his website out: # https://personal.sron.nl/~pault/ and its associated python package.

IMPORTANT WARNING:
In order to keep matplotlib's pie drawing being minimally janky, the size of the figure
needs to scale with the number of bins; it does this automatically.
If you ask for a lot of bins, go for a lower DPI to avoid overflowing your RAM
and crashing your computer; the default DPI should be ok (100) for most purposes (up to 100 x 100 bins)

basic example:
import scatterpie as sp
x,y,tags=sp.testdata() # generate some labelled data
sp.scatterpie(x,y,tags) # make the scatterpie


see main function scatterpie() for more functionality
"""
def testdata(tags_unique=['hello','world',],n=1000,xbounds=[-10,10],ybounds=[-5,5],p_tags=['default']):
    """
    Generate some data bounded by xbounds and ybounds, with each assigned a tag from tags_unique
    according to the probability ratios given in p_tags (default: equal probability).
    
    Inputs:
    tags_unique: labels for each unique 'tag' or classification for each data point.
    n: total number of data points to be generated.
    xbounds, ybounds: bounding values for x and y coordinates respectively..
    p_tags: relative probaility ratios for tags_unique (i.e. must be the same size!).
    e.g. tags_unique=[a,b,c] and p_tags=[1,2,3] means a ratio of 1:2:3 for each entry
    of tags_unique, so a probability of 1/6 for a, 2/6 for b, 3/6 for c. defaults to equal probability for each tag.
    
    Output:
    x: generated x coordinates.
    y: generated y coordinates.
    tags: the assigned tag for each x and y coordinate.

    """
    x=np.random.uniform(low=np.min(xbounds),high=np.max(xbounds),size=n)
    y=np.random.uniform(low=np.min(ybounds),high=np.max(ybounds),size=n)
    
    if p_tags==['default']:
        p_tags=[1]*len(tags_unique)#probabilities of each entry

    #normalise p_tags:
    p_tags=np.array(p_tags)/np.sum(p_tags)

    tags=np.random.choice(tags_unique,n,p=p_tags)


    return x,y,tags




nb.njit()
def setup_scatterpie(x,y,tags,weights='default',binx=10,biny=10):
    """
    Returns the pie centroids, ratios, and binning properties given x and y coordinates along with a list of tags for each coordinate.
    
    Inputs:
    x,y: lists of the x and y coordinates of the data, respectively.
    tags: list of tags (list of strings) associated with each data point.
    weights: relative weight of each data point. Defaults to all points being weighted equally.
    binx, biny: number of x and y bins, respectively. (total number of bins is binx * biny)
    
    Outputs:
    ratios: list of arrays with the ratio (for each tag) relevant for each pie chart to be plotted.
    centerx, centery: the coordinates of the centers of each pie chart to be plotted.
    binedgesx, bindedgesy: the bounds of each bin that were used to generate the ratios for each pie chart to be plotted.
"""
    x=np.array(x)
    y=np.array(y)
    tags=np.array(tags)


    minx=np.min(x)
    maxx=np.max(x)
    miny=np.min(y)
    maxy=np.max(y)
    rangex=maxx-minx
    rangey=maxy-miny
    binwidthx=rangex/binx
    binwidthy=rangey/biny

    if weights=='default':
        weights=np.ones(len(tags))

    tags_unique=np.unique(tags)


    binedgesx=np.arange(minx,maxx+binwidthx,rangex/binx)
    binedgesy=np.arange(miny,maxy+binwidthy,rangey/biny)

    centerx=[]
    centery=[]
    ratios=[]
    for i in np.arange(len(binedgesx)-1):
        for j in np.arange(len(binedgesy)-1):
            left=binedgesx[i]
            right=binedgesx[i+1]
            bottom=binedgesy[j]
            top=binedgesy[j+1]

            centerx.append((left+right)/2)
            centery.append((bottom+top)/2)
            
            bin_selector=((x>left) & (x<right) & (y>bottom) & (y<top))

            tagsbin=tags[bin_selector]
            weightsbin=weights[bin_selector]

            ratiostemp=np.zeros(len(tags_unique))
            for k,tag in enumerate(tags_unique):
                tag_selector=tagsbin==tag
                ratiostemp[k]=np.sum(weightsbin[tag_selector])
            ratios.append(ratiostemp)

    
    return ratios, centerx, centery, binedgesx, binedgesy 

def mscatter(x,y,ax=None, m=None, **kw):
    # custom scatterplot function capable of dealing with lists of markers in a single call, see solution from:
    # https://github.com/matplotlib/matplotlib/issues/11155
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def draw_pie(dist,xpos,ypos,size,ax=None,
    facecolorlist=['#77AADD', '#EE8866', '#EEDD88','#FFAABB', '#99DDFF','#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#000000']):
    """
    Code for the drawing of the pie charts, based on code originally written by Quang Hoang, see:
    https://stackoverflow.com/questions/56337732/how-to-plot-scatter-pie-chart-using-matplotlib
    
    Input:
    dist: ratio distributions (list of ratio arrays),
    xpos, ypos: coordinates of the pie chart centers
    size: size of the pie chart
    ax: esitisting axis object you want to use.
    facecolorlist: colors you want to use for each tag.
    """
    
    if ax is None:
        print('no axis passed')
        fig, ax = plt.subplots(figsize=(10,8))

    # for incremental pie slices
    xy=[]
    xposlist=[]
    yposlist=[]
    colorlist=[]
    for i,disttemp in enumerate(dist):
        cumsum = np.cumsum(disttemp)
        cumsum = cumsum/ cumsum[-1]
        pie = [0] + cumsum.tolist()
        # print(pie)

        j=0
        for r1, r2 in zip(pie[:-1], pie[1:]):
            if r1!=r2:
                angles = np.linspace(2 * np.pi * r1, 2 * np.pi * r2)
                x = [0] + np.cos(angles).tolist()
                y = [0] + np.sin(angles).tolist()
                xy.append(np.column_stack([x, y]))
                xposlist.append(xpos[i])
                yposlist.append(ypos[i])
                colorlist.append(facecolorlist[j])
            j+=1
    
    mscatter(xposlist, yposlist, m=xy, s=size,c=colorlist,rasterized=True)

def scatterpie(x,y,tags,weights='default',savepath='',savename='dummy.pdf',tags_unique_labels='default',
    binx=10,biny=10,dpi=100,piesize=2880,fontsize=24,ticksize=26,make_legend=True,xlabel='',ylabel='', close_all=True,
    facecolorlist= ['#77AADD', '#EE8866', '#EEDD88','#FFAABB', '#99DDFF','#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD', '#000000']):
    """
    Main function for making a scatterpie, a way of visuallising qualitativly distinct data in 2-D, fast.
    note: these figures tend to be quite large. This is so the piecharts don't become deformed and nasty.
    If the size of the figure becomes a problem, I recomend lowering the DPI before hacking into other aspects of the code.

    Inputs:
    x, y: lists of  x and y data coordinates of the data.
    tags: list of strings identifying each data coordinate. e.g. ['dog', 'duck', 'goose', 'duck', 'duck', ...]
    weights: weights of each data coordinate. defaults to equal weighting.
    savepath: path to the directory the resulting image will be saved.
    savename: name of the saved image, including extension.
    tags_unique: a list of all unique tags. only really useful if you want to use different
                tag labels (see tags_unique_labels) than the strings of the tags themselves.
    tags_unique_labels: labels for the tags. If you set this, you MUST also manually set tags_unique to make sure they line up!
    binx, biny: number of x and y bins, respectively. (total number of bins is binx * biny)
    dpi: resolution. lower to reduce figure size but reduce figure quality.
    facecolorlist: list of colors used for each differnent. Must have AT LEAST as entries colors as unqiue tags there are.
    piesize: sets the size of each pie chart
    fontsize: scales the font size for basically everything (different elements are set based on fixed offsets of this number.)
    ticksize: sets the size of the ax ticks.
    make_legend: boolean, True makes a legend for the tags, False turns the legend off.
    xlabel, ylabel: x and y labels.
    close_all: boolean, True closes the figure after its done (making the return worthless) while False does not.

    Outputs:
    fig,ax: figure and axis objects of the plot.
    """
    tinit=time.time()

    x=np.array(x)
    y=np.array(y)
    tags=np.array(tags)

    tags_unique=np.unique(tags)

    if tags_unique_labels=='default':
        tags_unique_labels=tags_unique

    print('setting up')
    ratios,centerx,centery,binedgesx,binedgesy=\
    setup_scatterpie(x,y,tags,weights=weights,binx=binx,biny=biny)

    print('done setting up')
    print(time.time()-tinit)
    print(fontsize)

    matplotlib.rcParams.update({'font.size':fontsize})
    matplotlib.rc('xtick', labelsize=fontsize-4) 
    matplotlib.rc('ytick', labelsize=fontsize-4)

    fig,ax=plt.subplots(figsize=(binx,biny),dpi=dpi)
    ax.tick_params('both',which='both',direction='in',length=fontsize*0.5,width=fontsize*0.05)

    draw_pie(ratios,centerx,centery,size=piesize,ax=ax,facecolorlist=facecolorlist)
    
    if make_legend==True:
        patchlist=[]
        for i,label in enumerate(tags_unique):
            patchlist.append(mpatches.Patch(color=facecolorlist[i], label=label))

        plt.legend(handles=patchlist,fontsize=fontsize+4,ncols=3,loc='lower center',bbox_to_anchor=(0.5,1.05))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout(pad=1.2)
    print('serving pies to:')
    print(savepath+savename)
    plt.savefig(savepath+savename)

    print(time.time()-tinit)
    if close_all:
        plt.close('all')
    return fig,ax

