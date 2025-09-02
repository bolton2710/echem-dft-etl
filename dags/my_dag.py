from airflow.sdk import dag, task
from datetime import timedelta, datetime
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.units import Bohr
from mp_api.client import MPRester
from pymatgen.io.vasp import Poscar
from pymatgen.core.surface import SlabGenerator
import subprocess
import matplotlib.image as mpimg
import psycopg2

#Key for Materials API
mykey="EWFzyhVIQyFk2EP1Q5ogndIAF2FH1rbp"

@task
def search_MaterialsAPI(elements, oxidation_state):
    """
    Call the Materials API to extract bulk materials
    Args:
        elements (list): elements to include, e.g. ["V-O", "V-O-H"]
        oxidation_state (list): oxidation state of any given element, e.g. ["V4+"]
    """
    #Run API to search for structure
    with MPRester(api_key=mykey) as mpr:
        docs=mpr.materials.oxidation_states.search(chemsys=elements, possible_species=oxidation_state)
    print(f'Number of entries from Materials Project: {len(docs)}')

    #Get structure from each entry
    bulks=[]
    ids=[]
    for entry in docs:
        ids.append(entry.material_id)
        bulks.append(entry.structure)

    #Write poscar to data folder
    poscar_dir=Path(__file__).resolve().parent.parent / "output" / "bulk_poscars"
    poscar_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0,len(bulks)):
        print('Saved bulk POSCAR', ids[i])
        Poscar(bulks[i]).write_file(f'{poscar_dir}/{ids[i]}.poscar') #save
    return ids #jsonalize-able

@task
def randomize_bulk(bulk_ids, nsample):
    """
    Randomly select a subset of bulk IDs.
    Args:
        bulk_ids (list): list of MP material IDs
        nsample (int): number of IDs to select
    """
    # Seed random() for reproducibility purpose
    random.seed(27)
    return random.sample(bulk_ids, min(nsample, len(bulk_ids)))

@task
def pymatgen_slab(ids, facets):
    """
    Call the Materials API to extract chosen bulk materials
    Use pymatgen SlabGenerator to cut slabs based on Miller's indices
    Args:
        ids (list): bulk MPRester ids
        facets (list): strings of Miller's indices, e.g., "111", "100"
    """
    #Run API to search for structure
    with MPRester(api_key=mykey) as mpr:
        docs=mpr.materials.summary.search(material_ids=ids)

    #Initialize folder to save poscars
    poscar_dir=Path(__file__).resolve().parent.parent / "output" / "slab_poscars"
    poscar_dir.mkdir(parents=True, exist_ok=True)
    poscar_dirs=[]

    #Loop through each bulk entry
    for entry in docs:
        #Get bulk structure and generate slab
        bulk=entry.structure
        for facet in facets:
            #Preprocess
            facetint=[int(i) for i in facet]
            slabgen = SlabGenerator(
                initial_structure=bulk,
                miller_index=facetint,
                min_slab_size=5.0,  # in Å
                min_vacuum_size=10.0,  # in Å
                center_slab=True,
                primitive=True,
                lll_reduce=False)
            slabs=slabgen.get_slabs(symmetrize=True)
            for i, slab in enumerate(slabs):
                #Save poscar
                save_dir=f'{poscar_dir}/{entry.material_id}-{facet}-{i}.poscar'
                print('Saved slab POSCAR', save_dir)
                Poscar(slab).write_file(save_dir) #save
                #Store directory to poscar
                poscar_dirs.append(save_dir)
    return poscar_dirs

@task
def randomize_slab(poscar_dirs):
    """
    Randomly select one of slab .
    Args:
        poscar_dirs (list): list of directory to poscar of slab
    """
    # Seed random() for reproducibility purpose
    random.seed(20)
    return random.sample(poscar_dirs, 1)[0]

from utils import write_jdftx #import util func to write input for dft
@task(execution_timeout=timedelta(seconds=10800)) #set 3hr max run time
def run_gcdft(poscar_path):
    filename=Path(poscar_path).stem
    #Create directory to run DFT
    dftdir=Path(__file__).resolve().parent.parent / "output" / "gc_dft" 
    dftdir.mkdir(parents=True, exist_ok=True)
    #For each surface, setup DFT
    atoms=read(poscar_path)
    for charge in [-0.1, 0.0, 0.1]:
        #Write input
        jinput=write_jdftx(atoms, charge)
        f=open(f"{dftdir}/{filename}_{charge}.in", "w")
        f.write(jinput)
        f.close()
        #Run jdftx
        subprocess.run(f"jdftx -i {dftdir}/{filename}_{charge}.in | tee {dftdir}/{filename}_{charge}.out", shell=True, cwd=dftdir)
        subprocess.run("rm fillings", shell=True, cwd=dftdir) #prevent intialize wrong number of electrons
    subprocess.run("rm wfns fluidState eigenvals", shell=True, cwd=dftdir) #clean up
    return filename

@task
def analyze_electrochem(filename):
    #Read GC-DFT output
    dftdir=Path(__file__).resolve().parent.parent / "output" / "gc_dft"
    fermis, nes=[], []
    for charge in [-0.1, 0.0, 0.1]:
        outfile=f"{dftdir}/{filename}_{charge}.out"
        with open(outfile) as f:
            for line in f:
                if "FillingsUpdate:  mu:" in line:
                    fermi=line.split()[2]
                    ne=line.split()[4]
        fermis.append(fermi)
        nes.append(ne)
    fermis=np.array(fermis).astype('float')
    nes=np.array(nes).astype('float')
    #Find PZC
    pzc=fermis[1]*-27.2114 - 4.66 #SHE
    #Find capacitance
    poscar_dir=Path(__file__).resolve().parent.parent / "output" / "slab_poscars"
    atoms=read(f"{poscar_dir}/{filename}.poscar")
    area=atoms.cell[0,0]*atoms.cell[1,1]*Bohr**2*1E-16 #cm2
    rhoes=-(nes-nes[1])/area*1.60217663E-19*1e6/2 #uC/cm^2 divided by 2 surfaces
    pots=fermis*-27.2114 - 4.66
    fit1d=np.polyfit(pots, rhoes, 1)
    #Make and save plot
    plt.figure(figsize=(5,3))
    plt.plot(pots, rhoes,'o',c='k')
    plt.plot(pots, pots*fit1d[0] + fit1d[1], ':', c='r')
    plt.ylabel(r'$\sigma_e$ ($\mu$C/cm$^2$)',fontsize=14)
    plt.xlabel(r'$\phi$ (V vs. SHE)',fontsize=14)
    plt.tick_params(axis='both',labelsize=13)
    plt.annotate(f'Capacitance: {fit1d[0]:.2f} $\mu$F/cm$^2$',xy=[0.1, 0.9],xycoords="axes fraction",c='r',fontsize=12)
    plt.annotate(f'PZC: {pzc:.3f} V vs. SHE',xy=[0.1, 0.8],xycoords="axes fraction",c='r',fontsize=12)
    plt.title(f"{filename}", fontsize=13)
    savedir=Path(__file__).resolve().parent.parent / "output" / "visualize"
    plt.savefig(f"{savedir}/{filename}_echem", dpi=300, bbox_inches="tight")
    return filename, pzc, fit1d[0]

@task
def load_db(results):
    #connection to postgres db
    conn=psycopg2.connect(host="my_postgres", dbname="echem_postgres", user="airflow", password="airflow")
    cur=conn.cursor()
    #Create table
    cur.execute("CREATE TABLE IF NOT EXISTS dft_echem (id SERIAL PRIMARY KEY, MP_id VARCHAR(100) UNIQUE, pzc FLOAT, capacitance FLOAT);")
    #Insert data
    cur.execute("INSERT INTO dft_echem (MP_id, pzc, capacitance) VALUES (%s, %s, %s) ON CONFLICT (MP_id) DO NOTHING;", 
                (results[0], results[1], results[2]))
    #Check table
    cur.execute("SELECT * FROM dft_echem;")
    print(cur.fetchone())
    #Commit changes
    conn.commit()
    #Close-out
    cur.close()
    conn.close()
    return results[0]

@task
def write_report(results):
    filename=results[0]
    #save atomic structure as png
    poscar_dir=Path(__file__).resolve().parent.parent / "output" / "slab_poscars"
    atoms=read(f"{poscar_dir}/{filename}.poscar")
    savedir=Path(__file__).resolve().parent.parent / "output" / "visualize" 
    savedir.mkdir(parents=True, exist_ok=True)
    write(f'{savedir}/{filename}_struct.png', atoms, rotation='90x')
    #Combine
    echem_img=mpimg.imread(f'{savedir}/{filename}_echem.png')
    struct_img=mpimg.imread(f'{savedir}/{filename}_struct.png')
    fig, ax=plt.subplots(figsize=(6, 4))
    ax.imshow(echem_img)
    ax.axis("off")  # remove axes
    inset_ax=fig.add_axes([0.55, 0.3, 0.4, 0.4])  
    inset_ax.imshow(struct_img)
    inset_ax.axis("off")
    plt.savefig(f'{savedir}/{filename}', bbox_inches="tight", dpi=150)

    #Write report as simple markdown file:
    report=f"\n\n![](visualize/{filename}.png)"
    writedir=Path(__file__).resolve().parent.parent / "output"
    f=open(f"{writedir}/report.md", "a")
    f.write(report)

#----------DAG settings---------------
@dag(
    dag_id="MP-pymatgen-jDFTx",
    start_date=datetime(2025, 8, 10),
    schedule=None,
    catchup=False,
    default_args={"retries": 0},
)
def run_dag():
    """
    A workflow for generating V-O surfaces and running GC-DFT.
    """
    bulk_ids=search_MaterialsAPI(["V-O"], ["V4+"])
    randomized_ids=randomize_bulk(bulk_ids, 2)
    poscar_dirs=pymatgen_slab(randomized_ids, ["111"])
    randomized_slab=randomize_slab(poscar_dirs)
    sample=run_gcdft(randomized_slab)
    results=analyze_electrochem(sample)
    load_db(results)
    write_report(results)
run_dag()