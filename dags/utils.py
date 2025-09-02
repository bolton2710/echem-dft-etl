from ase.units import Bohr

def write_jdftx(atoms, charge):
    #Create JDFTx input
    jinput=''
    #Add lattice info
    R = atoms.get_cell() / Bohr #convert A to Bohr
    jinput += 'lattice \\\n'
    for i in range(3):
        for j in range(3):
            jinput += '%f  ' % (R[j, i])
        if(i != 2):
            jinput += '\\'
        jinput += '\n'
    #Add ion info
    atomPos = [x / Bohr for x in list(atoms.get_positions())]  # Also convert to bohr
    atomNames = atoms.get_chemical_symbols()   # Get element names in a list
    jinput += '\ncoords-type cartesian\n'
    #Check for constraints
    if atoms.constraints!=[] :cons=atoms.constraints[0].get_indices()
    else : cons=[]
    for i in range(len(atomPos)):
        if i in cons: atomFix=0
        else: atomFix=1
        jinput += 'ion %s %f %f %f %i\n' % (atomNames[i], atomPos[i][0], atomPos[i][1], atomPos[i][2],atomFix)
    #DFT settings
    dft_settings=["#DFT settings\n",
        "dump End State\n",
        "dump-name $VAR\n",
        "initial-state $VAR\n",
        "coulomb-interaction Periodic\n",
        "ion-species GBRV/$ID_pbe.uspp\n",
        "elec-cutoff 5\n",
        "spintype no-spin\n",
        "symmetries automatic\n",
        "elec-ex-corr gga-PBE\n",
        "kpoint 0.5 0.5 0.5 1\n",
        "kpoint-folding 1 1 1\n",
        "lcao-params 15 5E-5\n",
        "elec-smearing Fermi 0.01\n",
        "electronic-minimize nIterations 30 energyDiffThreshold 1E-5\n",
        "fluid LinearPCM #solvation model\n",
        "pcm-variant CANDLE\n",
        "fluid-cation Na+ 1.\n",
        "fluid-anion F- 1.\n",
        f"elec-initial-charge {charge}"]
    for line in dft_settings: jinput+=line
    return jinput