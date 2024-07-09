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

    #'fix' for if you feed it data that is all at the same point in one or both axes.
    #it being a shit number is necessary to avoid a bin edge falling right on your data!
    if minx==maxx:
        minx+=-0.099236423413
        maxx+=0.099236423413

    if miny==maxy:
        miny+=-0.099236423413
        maxy+=0.099236423413


    if minx==maxx:
        minx+=-0.1
        maxx+=0.1

    if miny==maxy:
        miny+=-0.1
        maxy+=0.1

    
        
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

            # the annoying precision modifications terms are due to machine precision issues
            # that can have an annoying effect when using grid-based x and y data that affect the edges.
            # There is almost certainly a better way to do this.
            if i==0:
                left-=(1e-6*left)
                # print('left edge!' + str(left))

            if j==0:
                bottom-=(1e-6*bottom)
                # print('bottom edge!' + str(bottom))


            if i+1==len(binedgesx)-1:
                right+=(1e-6*right)
                # print('right edge!' + str(right))

            if j+1==len(binedgesy)-1:
                top+=(1e-6*top)
                # print('top edge: ' + str(top))
            centerx.append((left+right)/2)
            centery.append((bottom+top)/2)
            
            bin_selector=((x>left) & (x<=right) & (y>bottom) & (y<=top))

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
    ax: existing axis object you want to use.
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
    
    mscatter(xposlist, yposlist, m=xy, s=size,c=colorlist,ax=ax,rasterized=True)

def scatterpie(x,y,tags,weights='default',savepath='',savename='dummy.pdf',savefig=True,tags_unique_labels='default',
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

    if savefig==True:
        print('serving pies to:')
        print(savepath+savename)
        plt.savefig(savepath+savename,bbox_inches='tight')

    print(time.time()-tinit)
    if close_all:
        plt.close('all')
    return fig,ax

def fancy_scatterpie(x,y,tags,weights='default',savepath='',savename='dummy.pdf',savefig=True,tags_unique_labels='default',
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


    #setup for the stacked claw hists
    hist_arr_x=[]
    hist_arr_y=[]
    hist_arr_weights=[]

    for i,tag in enumerate(tags_unique):
        if weights=='default':
            weights=np.ones(len(x))
        temp_bool=tags==tag
        hist_arr_x.append(x[temp_bool])
        hist_arr_y.append(y[temp_bool])
        hist_arr_weights.append(weights[temp_bool])

    ax_bounds_x=[np.min(x)-0.05*np.max(x),np.max(x)+0.05*np.max(x)]
    ax_bounds_y=[np.min(y)-0.05*np.max(y),np.max(y)+0.05*np.max(y)]
    print('done setting up')
    print(time.time()-tinit)
    print(fontsize)

    matplotlib.rcParams.update({'font.size':fontsize})
    matplotlib.rc('xtick', labelsize=fontsize-4) 
    matplotlib.rc('ytick', labelsize=fontsize-4)




    fig=plt.figure(num=None,figsize=(20,20),dpi=dpi,facecolor='w',edgecolor='k')
    widths=[3,1]
    heights=[1,3]
    spec=fig.add_gridspec(ncols=2,nrows=2,width_ratios=widths,height_ratios=heights)
    # ax=plt.subplot(223)
    ax=fig.add_subplot(spec[1,0])

    ax.tick_params('both',which='both',direction='in',length=fontsize*0.5,width=fontsize*0.05)

    draw_pie(ratios,centerx,centery,size=piesize,ax=ax,facecolorlist=facecolorlist)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(ax_bounds_x)
    ax.set_ylim(ax_bounds_y)
    
    if make_legend==True:
        ax=fig.add_subplot(spec[0,1])
        ax.set_axis_off()
        patchlist=[]
        for i,label in enumerate(tags_unique):
            patchlist.append(mpatches.Patch(color=facecolorlist[i], label=label))

        plt.legend(handles=patchlist,fontsize=fontsize+4,ncols=2,loc='upper center',bbox_to_anchor=(0.5,0.95))

    


    
    ax=fig.add_subplot(spec[0,0])
    ax.hist(hist_arr_x,weights=hist_arr_weights,bins=binx,
        color=facecolorlist[:len(tags_unique)],stacked=True,linewidth=1,edgecolor='k',rasterized=True)
    ax.set_xlim(ax_bounds_x)

    ax=fig.add_subplot(spec[1,1])
    ax.hist(hist_arr_y,weights=hist_arr_weights,bins=biny,orientation='horizontal',
        color=facecolorlist[:len(tags_unique)],stacked=True,linewidth=1,edgecolor='k',rasterized=True)
    ax.set_ylim(ax_bounds_y)

    if savefig==True:
        print('serving pies to:')
        print(savepath+savename)
        plt.savefig(savepath+savename,bbox_inches='tight')

    print(time.time()-tinit)
    if close_all:
        plt.close('all')
    return fig,ax