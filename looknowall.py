import numpy as np
import matplotlib.pyplot as plt

def look(obs, model=None, jobn=None, save=0, savepath='', colkeys=None, diskcomb=0, xlim=[2e-1, 2e3], ylim=[1e-15, 1e-9], params=1, leg=1, public=0):
    """
    Creates a plot of a model and the observations for a given target.
    
    INPUTS
    model: The object containing the target's model. Should be an instance of the TTS_Model class. This is an optional input.
    obs: The object containing the target's observations. Should be an instance of the TTS_Obs class.
    jobn: The "job number." This is meaningless for observation-only plots, but if you save the file, we require a number.
    save: BOOLEAN -- If 1 (True), will save the plot in a pdf file. If 0 (False), will output to screen.
    savepath: The path that a saved PDF file will be written to. This is defaulted to the hard-coded figurepath at top of this file.
    colkeys: An optional input array of color strings. This can be used to overwrite the normal color order convention. Options include:
             p == purple, r == red, m == magenta, b == blue, c == cyan, l == lime, t == teal, g == green, y == yellow, o == orange,
             k == black, w == brown, v == violet, d == gold, n == pumpkin, e == grape, j == jeans, s == salmon
             If not specified, the default order will be used, and once you run out, we'll have an error. So if you have more than 18
             data types, you'll need to supply the order you wish to use (and which to repeat). Or you can add new colors using html tags
             to the code, and then update this header.
    diskcomb: BOOLEAN -- If 1 (True), will combine outer wall and disk components into one for plotting. If 0 (False), will separate.
    xlim: A list containing the lower and upper x-axis limits, respectively. Has default values.
    ylim: A list containing the lower and upper y-axis limits, respectively. Has default values.
    params: BOOLEAN -- If 1 (True), the parameters for the model will be printed on the plot.
    leg: BOOLEAN -- If 1 (True), the legend will be printed on the plot.
    
    OUTPUT
    A plot. Can be saved or plotted to the screen based on the "save" input parameter.
    """

    photkeys            = obs.photometry.keys()         # obs.photometry and obs.spectra are dictionaries.
    speckeys            = obs.spectra.keys()
    colors              = {'p':'#7741C8', 'r':'#F50C0C', 'm':'#F50CA3', 'b':'#2B0CF5', 'c':'#0CE5F5', 'l':'#33F50C', 't':'#4DCE9B', \
                           'g':'#1D5911', 'y':'#BFB91E', 'o':'#F2A52A', 'k':'#060605', 'w':'#5A3A06', 'v':'#BD93D2', 'd':'#FFD900', \
                           'n':'#FF7300', 'e':'#9A00FA', 'j':'#00AAFF', 's':'#D18787'}
    if colkeys == None:
        colkeys         = ['p', 'r', 'o', 'b', 'c', 'm', 'g', 'y', 'l', 'k', 't', 'w', 'v', 'd', 'n', 'e', 'j', 's']    # Order in which colors are used

    # Let the plotting begin!
    if save == 0:
        plt.clf()
    plt.figure(1)
    
    # Plot the spectra first:
    for sind, skey in enumerate(speckeys):
        plt.plot(obs.spectra[skey]['wl'], obs.spectra[skey]['lFl'], color=colors[colkeys[sind]] , linewidth=2.0, label=skey)
    
    # Next is the photometry:
    for pind, pkey in enumerate(photkeys):
        # If an upper limit only:
        if pkey in obs.ulim:
            plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'v', \
                     color=colors[colkeys[pind+len(speckeys)]], markersize=7, label=pkey, zorder=pind+10)
        # If not an upper limit, plot as normal:
        else:
            if 'err' not in obs.photometry[pkey].keys():
                plt.plot(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], 'o', mfc='w', mec=colors[colkeys[pind+len(speckeys)]], mew=1.0,\
                         markersize=7, label=pkey, zorder=pind+10)
            else:
                plt.errorbar(obs.photometry[pkey]['wl'], obs.photometry[pkey]['lFl'], yerr=obs.photometry[pkey]['err'], \
                             mec=colors[colkeys[pind+len(speckeys)]], fmt='o', mfc='w', mew=1.0, markersize=7, \
                             ecolor=colors[colkeys[pind+len(speckeys)]], elinewidth=2.0, capsize=3.0, label=pkey, zorder=pind+10)
    # Publication style?
    if public:
        # Now, the model (if a model supplied):
        if model != None:
            modkeys         = model.data.keys()
            if 'phot' in modkeys:
                plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
            # Will be combining the inner/outer walls with the disk emission component:
            if 'dust' in modkeys:
                if 'owall' in modkeys:
                    if 'newIWall' in model.__dict__:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.newIWall + model.data['disk'] + model.data['dust']
                        else:
                            diskflux = model.newIWall + model.data['disk'] + model.data['dust']
                    else:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.data['iwall']+ model.data['disk'] + model.data['dust']
                        else:
                            diskflux = model.data['iwall'] + model.data['disk'] + model.data['dust']
                else:
                    if 'newIWall' in model.__dict__:
                        diskflux = model.newIWall + model.data['disk'] + model.data['dust']
                    else:
                        diskflux = model.data['iwall'] + model.data['disk'] + model.data['dust']
            else:
                if 'owall' in modkeys:
                    if 'newIWall' in model.__dict__:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.newIWall + model.data['disk']
                        else:
                            diskflux = model.newIWall + model.data['disk']
                    else:
                        if 'newOWall' in model.__dict__:
                            diskflux = model.data['iwall']+ model.data['disk']
                        else:
                            diskflux = model.data['iwall'] + model.data['disk']
                else:
                    if 'newIWall' in model.__dict__:
                        diskflux = model.newIWall + model.data['disk']
                    else:
                        diskflux = model.data['iwall'] + model.data['disk']
            plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
            if 'scatt' in modkeys:
                plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')
            if 'shock' in modkeys:
                plt.plot(model.data['WTTS']['wl'], model.data['WTTS']['lFl'], c='b', linewidth=2.0, zorder=1, label='WTTS Photosphere')
                plt.plot(model.data['shock']['wl'], model.data['shock']['lFl'], c=colors['j'], linewidth=2.0, zorder=2, label='MagE')
                plt.plot(model.data['shockLong']['wl'], model.data['shockLong']['lFl'], c=colors['s'], linewidth=2.0, zorder=2, label='Shock Model')
            if 'total' in modkeys:
                plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
    else:
        # Now, the model (if a model supplied):
        if model != None:
            modkeys         = model.data.keys()
            if 'phot' in modkeys:
                plt.plot(model.data['wl'], model.data['phot'], ls='--', c='b', linewidth=2.0, label='Photosphere')
            if 'owall' in modkeys:
                try:
                    plt.plot(model.data['wl'], model.newIWall, ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')
                except AttributeError:
                    if 'iwall' in modkeys:
                        plt.plot(model.data['wl'], model.data['iwall'], ls='--', c='#53EB3B', linewidth=2.0, label='Inner Wall')
            else:
                try:
                    plt.plot(model.data['wl'], model.newIWall, ls='--', c='#53EB3B', linewidth=2.0, label='Wall')
                except AttributeError:
                    if 'iwall' in modkeys:
                        plt.plot(model.data['wl'], model.data['iwall'], ls='--', c='#53EB3B', linewidth=2.0, label='Wall')
            if diskcomb:
                try:
                    diskflux     = model.newOwall + model.data['disk']
                except AttributeError:
                    try:
                        diskflux = model.data['owall'] + model.data['disk']
                    except KeyError:
                        print('LOOK: Error, tried to combine outer wall and disk components but one component is missing!')
                    else:    
                        plt.plot(model.data['wl'], diskflux, ls='--', c='#8B0A1E', linewidth=2.0, label='Outer Disk')
            else:
                try:
                    plt.plot(model.data['wl'], model.newOWall, ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
                except AttributeError:
                    if 'owall' in modkeys:
                        plt.plot(model.data['wl'], model.data['owall'], ls='--', c='#E9B021', linewidth=2.0, label='Outer Wall')
                if 'disk' in modkeys:
                    plt.plot(model.data['wl'], model.data['disk'], ls='--', c='#8B0A1E', linewidth=2.0, label='Disk')
            if 'dust' in modkeys:
                plt.plot(model.data['wl'], model.data['dust'], ls='--', c='#F80303', linewidth=2.0, label='Opt. Thin Dust')
            if 'scatt' in modkeys:
                plt.plot(model.data['wl'], model.data['scatt'], ls='--', c='#7A6F6F', linewidth=2.0, label='Scattered Light')
            if 'shock' in modkeys:
                plt.plot(model.data['WTTS']['wl'], model.data['WTTS']['lFl'], c='b', linewidth=2.0, zorder=1, label='WTTS Photosphere')
                plt.plot(model.data['shock']['wl'], model.data['shock']['lFl'], c=colors['j'], linewidth=2.0, zorder=2, label='MagE')
                plt.plot(model.data['shockLong']['wl'], model.data['shockLong']['lFl'], c=colors['s'], linewidth=2.0, zorder=2, label='Shock Model')
            if 'total' in modkeys:
                plt.plot(model.data['wl'], model.data['total'], c='k', linewidth=2.0, label='Combined Model')
    # Now, the relevant meta-data:
    if model != None:    
        if params:
            plt.figtext(0.60,0.88,'Eps = '+ str(model.eps), color='#010000', size='9')
            plt.figtext(0.80,0.88,'Alpha = '+ str(model.alpha), color='#010000', size='9')
            plt.figtext(0.60,0.82,'Amax = '+ str(model.amax), color='#010000', size='9')
            plt.figtext(0.60,0.85,'Rin = '+ str(model.rin), color='#010000', size='9')
            plt.figtext(0.80,0.85,'Rout = '+ str(model.rdisk), color='#010000', size='9')
            plt.figtext(0.60,0.79,'Altinh = '+ str(model.wallH), color='#010000', size='9')
            plt.figtext(0.80,0.82,'Mdot = '+ str(model.mdot), color='#010000', size='9')
            # If we have an outer wall height:
            try:
                plt.figtext(0.80,0.79,'AltinhOuter = '+ str(model.owallH), color='#010000', size='9')
            except AttributeError:
                plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.temp), color='#010000', size='9')
            else:
                plt.figtext(0.60,0.76,'IWall Temp = '+ str(model.itemp), color='#010000', size='9')
                plt.figtext(0.80,0.76,'OWall Temp = '+ str(model.temp), color='#010000', size='9')
        
    # Lastly, the remaining parameters to plotting (mostly aesthetics):
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.ylabel(r'${\rm \lambda F_{\lambda}\; (erg\; s^{-1}\; cm^{-2})}$')
    plt.xlabel(r'${\rm {\bf \lambda}\; (\mu m)}$')
    #plt.title(obs.name.upper())
    if leg:
        plt.legend(loc=3)
    
    # Should we save or should we plot?
    if save:
        if type(jobn) != int:
            raise ValueError('LOOK: Jobn must be an integer if you wish to save the plot.')
        jobstr          = str(jobn).zfill(3)
        plt.savefig(savepath + obs.name.upper() + '_' + jobstr + '.pdf', dpi=250)
        plt.clf()
    else:
        plt.show()

    return