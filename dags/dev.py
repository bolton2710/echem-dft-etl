from airflow.sdk import dag, task
from datetime import timedelta, datetime
from pathlib import Path
import numpy as np
from ase.units import Bohr
from ase.io import read, write
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import psycopg2

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
    dag_id="dev",
    start_date=datetime(2025, 8, 10),
    schedule=None,
    catchup=False,
    default_args={"retries": 0},
)
def dev():
    print("Dev analysis")
    results=analyze_electrochem("mp-755394-111-3")
    load_db(results)
    write_report(results)
dev()