"""
A function to replicate the vacuum swing adsorption calculation performed by Boyd et al. 
in "Data-driven design of metal-organic frameworks for wet flue gas CO2 capture"
https://archive.materialscloud.org/record/2018.0016/v3

Mixture adsorption was simulated with the conditions 298K and 0.15:0.85 CO2/N2 
with a total pressure of 1 bar. The data file reports working capacities, which is 
the difference of adsorption of CO2 between two thermodynamic state points.
two desorption values were simulated; 0.1 bar CO2 at 363K (vacuum swing adsorption)
and 0.7 bar CO2 at 413K (temperature swing adsorption).
"""
import random
import numpy as np
from mofdiff.common.sys_utils import timeout
from mofdiff.gcmc import gcmc_wrapper
import re


def extract_raspa_output(raspa_output, has_N2=False):
    final_loading_section = re.findall(
        r"Number of molecules:\n=+[^=]*(?=)", raspa_output
    )[0]
    enthalpy_of_adsorption_section = re.findall(
        r"Enthalpy of adsorption:\n={2,}\n(.+?)\n={2,}", raspa_output, re.DOTALL
    )[0]

    CO2_subsection = re.findall(
        r"Component \d \[CO2\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
    )[0]
    adsorbed_CO2 = float(
        re.findall(
            r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
            CO2_subsection,
        )[0]
    )

    if has_N2:
        CO2_enthalpy_subsection = re.findall(
            r"\[CO2\].*?(?=component|\Z)", enthalpy_of_adsorption_section, re.DOTALL
        )[0]

        enthalpy_of_adsorption_CO2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", CO2_enthalpy_subsection)[0]
        ) * 0.239
        heat_of_adsorption_CO2 = -1 * enthalpy_of_adsorption_CO2

        N2_subsection = re.findall(
            r"Component \d \[N2\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
        )[0]
        adsorbed_N2 = float(
            re.findall(
                r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
                N2_subsection,
                re.DOTALL,
            )[0]
        )
        CO2_N2_selectivity = adsorbed_CO2 / adsorbed_N2

        N2_enthalpy_subsection = re.findall(
            r"\[N2\].*?(?=component|\Z)", enthalpy_of_adsorption_section, re.DOTALL
        )[0]
        enthalpy_of_adsorption_N2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", N2_enthalpy_subsection)[0]
        ) * 0.239
        heat_of_adsorption_N2 = -1 * enthalpy_of_adsorption_N2

        return (
            adsorbed_CO2,
            adsorbed_N2,
            CO2_N2_selectivity,
            heat_of_adsorption_CO2,
            heat_of_adsorption_N2,
        )

    else:
        CO2_enthalpy_subsection = re.findall(
            r"Total enthalpy of adsorption.*?(?=Q=-H|\Z)",
            enthalpy_of_adsorption_section,
            re.DOTALL,
        )[0]
        enthalpy_of_adsorption_CO2 = float(
            re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", CO2_enthalpy_subsection)[0]
        ) * 0.239 # (kcal per mol)
        heat_of_adsorption_CO2 = -1 * enthalpy_of_adsorption_CO2

        return adsorbed_CO2, heat_of_adsorption_CO2

@timeout(36000)
def working_capacity_vacuum_swing(cif_file, calc_charges=True,
                                  rundir='./temp', rewrite_raspa_input=False,
                                  charge_method="eqeq"):
    random.seed(4)
    np.random.seed(4)
    # adsorption conditions
    adsorbed = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2", "N2"],
        sorbates_mol_fraction=[0.15, 0.85],
        temperature=298,
        pressure=100000,  # 1 bar
        rundir=rundir,
    )

    if calc_charges:
        gcmc_wrapper.calculate_charges(adsorbed, method=charge_method)
    gcmc_wrapper.run_gcmc_simulation(
        adsorbed,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    (
        adsorbed_CO2,
        adsorbed_N2,
        CO2_N2_selectivity_298,
        heat_of_adsorption_CO2_298,
        heat_of_adsorption_N2_298,
    ) = extract_raspa_output(adsorbed.raspa_output, has_N2=True)

    # desorption conditions
    residual = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["CO2"],
        sorbates_mol_fraction=[1],
        temperature=363,  # 363,
        pressure=10000,  # 10000 # 0.1 bar
        rundir=rundir,
    )
    
    if calc_charges:
        gcmc_wrapper.calculate_charges(residual, method=charge_method)
    gcmc_wrapper.run_gcmc_simulation(
        residual,
        rewrite_raspa_input=rewrite_raspa_input,
    )

    residual_CO2, heat_of_adsorption_CO2_363 = extract_raspa_output(
        residual.raspa_output, has_N2=False
    )

    output = {
        "file": str(cif_file),
        "working_capacity_vacuum_swing": adsorbed_CO2 - residual_CO2,
        "CO2_N2_selectivity": CO2_N2_selectivity_298,
        "CO2_uptake_P0.15bar_T298K": adsorbed_CO2,
        "CO2_uptake_P0.10bar_T363K": residual_CO2,
        "CO2_heat_of_adsorption_P0.15bar_T298K": heat_of_adsorption_CO2_298,
        "CO2_heat_of_adsorption_P0.10bar_T363K": heat_of_adsorption_CO2_363,
        "N2_uptake_P0.85bar_T298K": adsorbed_N2,
        "N2_heat_of_adsorption_P0.85bar_T298K": heat_of_adsorption_N2_298,
    }

    return output

def run_or_fail(cif_path):
    try:
        return working_capacity_vacuum_swing(cif_path)
    except Exception as e:
        print(e)
        return None


def extract_raspa_output_nh3(raspa_output):
    """Extract NH3 adsorption data from RASPA output."""
    final_loading_section = re.findall(
        r"Number of molecules:\n=+[^=]*(?=)", raspa_output
    )[0]
    enthalpy_of_adsorption_section = re.findall(
        r"Enthalpy of adsorption:\n={2,}\n(.+?)\n={2,}", raspa_output, re.DOTALL
    )[0]

    NH3_subsection = re.findall(
        r"Component \d \[NH3\].*?(?=Component|\Z)", final_loading_section, re.DOTALL
    )[0]
    
    # Get loading in mol/kg
    adsorbed_NH3_mol_kg = float(
        re.findall(
            r"(?<=Average loading absolute \[mol/kg framework\])\s*\d*\.\d*",
            NH3_subsection,
        )[0]
    )
    
    # Get loading in mmol/g (which is same as mol/kg)
    adsorbed_NH3_mmol_g = adsorbed_NH3_mol_kg
    
    # Get loading in mg/g
    NH3_molar_mass = 17.031  # g/mol
    adsorbed_NH3_mg_g = adsorbed_NH3_mol_kg * NH3_molar_mass
    
    # Get enthalpy of adsorption
    NH3_enthalpy_subsection = re.findall(
        r"Total enthalpy of adsorption.*?(?=Q=-H|\Z)",
        enthalpy_of_adsorption_section,
        re.DOTALL,
    )[0]
    enthalpy_of_adsorption_NH3 = float(
        re.findall(r"(?<=\[K\])\s*-?\d*\.\d*", NH3_enthalpy_subsection)[0]
    ) * 0.239  # Convert to kcal/mol
    heat_of_adsorption_NH3 = -1 * enthalpy_of_adsorption_NH3

    return adsorbed_NH3_mol_kg, adsorbed_NH3_mmol_g, adsorbed_NH3_mg_g, heat_of_adsorption_NH3


@timeout(36000)
def nh3_uptake_simulation(cif_file, calc_charges=True, rundir='./temp', 
                          rewrite_raspa_input=False,
                          temperature=298, pressure=101325,
                          charge_method="eqeq"):
    """
    Run GCMC simulation to calculate NH3 uptake for a MOF structure.
    
    Args:
        cif_file: Path to the CIF file of the MOF structure
        calc_charges: Whether to calculate charges (default: True)
        rundir: Directory for intermediate files
        rewrite_raspa_input: Whether to rewrite RASPA input files
        temperature: Simulation temperature in K (default: 298 K)
        pressure: Simulation pressure in Pa (default: 101325 Pa = 1 bar)
        charge_method: Charge equilibration method ("eqeq", "mepo", "gasteiger")
    
    Returns:
        Dictionary with NH3 uptake results
    """
    random.seed(4)
    np.random.seed(4)
    
    # Create GCMC simulation object for NH3
    simulation = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["NH3"],
        sorbates_mol_fraction=[1.0],
        temperature=temperature,
        pressure=pressure,
        rundir=rundir,
    )
    
    # Calculate charges if requested
    if calc_charges:
        gcmc_wrapper.calculate_charges(simulation, method=charge_method)
    
    # Run GCMC simulation
    gcmc_wrapper.run_gcmc_simulation(
        simulation,
        rewrite_raspa_input=rewrite_raspa_input,
    )
    
    # Extract results
    (
        adsorbed_NH3_mol_kg,
        adsorbed_NH3_mmol_g,
        adsorbed_NH3_mg_g,
        heat_of_adsorption_NH3,
    ) = extract_raspa_output_nh3(simulation.raspa_output)
    
    output = {
        "file": str(cif_file),
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "NH3_uptake_mol_kg": adsorbed_NH3_mol_kg,
        "NH3_uptake_mmol_g": adsorbed_NH3_mmol_g,
        "NH3_uptake_mg_g": adsorbed_NH3_mg_g,
        "NH3_heat_of_adsorption_kcal_mol": heat_of_adsorption_NH3,
    }
    
    return output


@timeout(36000)
def nh3_isotherm_simulation(cif_file, calc_charges=True, rundir='./temp',
                             rewrite_raspa_input=False,
                             temperature=298,
                             pressures=None,
                             charge_method="eqeq"):
    """
    Run GCMC simulations at multiple pressures to generate NH3 adsorption isotherm.
    
    Args:
        cif_file: Path to the CIF file of the MOF structure
        calc_charges: Whether to calculate charges (default: True)
        rundir: Directory for intermediate files
        rewrite_raspa_input: Whether to rewrite RASPA input files
        temperature: Simulation temperature in K (default: 298 K)
        pressures: List of pressures in Pa (default: logarithmic range from 100 Pa to 101325 Pa)
    
    Returns:
        Dictionary with isotherm data
    """
    random.seed(4)
    np.random.seed(4)
    
    # Default pressure points for isotherm (100 Pa to 1 bar in log scale)
    if pressures is None:
        pressures = [100, 500, 1000, 5000, 10000, 50000, 101325]
    
    isotherm_data = []
    
    for pressure in pressures:
        try:
            result = nh3_uptake_simulation(
                cif_file,
                calc_charges=calc_charges,
                rundir=rundir,
                rewrite_raspa_input=rewrite_raspa_input,
                temperature=temperature,
                pressure=pressure,
                charge_method=charge_method,
            )
            isotherm_data.append(result)
            # Only calculate charges once
            calc_charges = False
        except Exception as e:
            print(f"Error at pressure {pressure} Pa: {e}")
            isotherm_data.append({
                "file": str(cif_file),
                "temperature_K": temperature,
                "pressure_Pa": pressure,
                "error": str(e)
            })
    
    output = {
        "file": str(cif_file),
        "temperature_K": temperature,
        "isotherm": isotherm_data,
    }
    
    return output


@timeout(36000)
def nh3_working_capacity(cif_file, calc_charges=True, rundir='./temp',
                          rewrite_raspa_input=False,
                          adsorption_temperature=298,
                          adsorption_pressure=101325,
                          desorption_temperature=373,
                          desorption_pressure=10000,
                          charge_method="eqeq"):
    """
    Calculate NH3 working capacity (difference between adsorption and desorption).
    
    Args:
        cif_file: Path to the CIF file of the MOF structure
        calc_charges: Whether to calculate charges (default: True)
        rundir: Directory for intermediate files
        rewrite_raspa_input: Whether to rewrite RASPA input files
        adsorption_temperature: Adsorption temperature in K (default: 298 K)
        adsorption_pressure: Adsorption pressure in Pa (default: 101325 Pa = 1 bar)
        desorption_temperature: Desorption temperature in K (default: 373 K = 100°C)
        desorption_pressure: Desorption pressure in Pa (default: 10000 Pa = 0.1 bar)
    
    Returns:
        Dictionary with working capacity results
    """
    random.seed(4)
    np.random.seed(4)
    
    # Adsorption conditions
    adsorbed = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["NH3"],
        sorbates_mol_fraction=[1.0],
        temperature=adsorption_temperature,
        pressure=adsorption_pressure,
        rundir=rundir,
    )
    
    if calc_charges:
        gcmc_wrapper.calculate_charges(adsorbed, method=charge_method)
    
    gcmc_wrapper.run_gcmc_simulation(
        adsorbed,
        rewrite_raspa_input=rewrite_raspa_input,
    )
    
    (
        adsorbed_NH3_mol_kg,
        adsorbed_NH3_mmol_g,
        adsorbed_NH3_mg_g,
        heat_of_adsorption_NH3_ads,
    ) = extract_raspa_output_nh3(adsorbed.raspa_output)
    
    # Desorption conditions
    residual = gcmc_wrapper.gcmc_simulation(
        cif_file,
        sorbates=["NH3"],
        sorbates_mol_fraction=[1.0],
        temperature=desorption_temperature,
        pressure=desorption_pressure,
        rundir=rundir,
    )
    
    if calc_charges:
        gcmc_wrapper.calculate_charges(residual, method=charge_method)
    
    gcmc_wrapper.run_gcmc_simulation(
        residual,
        rewrite_raspa_input=rewrite_raspa_input,
    )
    
    (
        residual_NH3_mol_kg,
        residual_NH3_mmol_g,
        residual_NH3_mg_g,
        heat_of_adsorption_NH3_des,
    ) = extract_raspa_output_nh3(residual.raspa_output)
    
    # Calculate working capacity
    working_capacity_mol_kg = adsorbed_NH3_mol_kg - residual_NH3_mol_kg
    working_capacity_mmol_g = adsorbed_NH3_mmol_g - residual_NH3_mmol_g
    working_capacity_mg_g = adsorbed_NH3_mg_g - residual_NH3_mg_g
    
    output = {
        "file": str(cif_file),
        "adsorption_temperature_K": adsorption_temperature,
        "adsorption_pressure_Pa": adsorption_pressure,
        "desorption_temperature_K": desorption_temperature,
        "desorption_pressure_Pa": desorption_pressure,
        "NH3_uptake_adsorption_mol_kg": adsorbed_NH3_mol_kg,
        "NH3_uptake_adsorption_mmol_g": adsorbed_NH3_mmol_g,
        "NH3_uptake_adsorption_mg_g": adsorbed_NH3_mg_g,
        "NH3_uptake_desorption_mol_kg": residual_NH3_mol_kg,
        "NH3_uptake_desorption_mmol_g": residual_NH3_mmol_g,
        "NH3_uptake_desorption_mg_g": residual_NH3_mg_g,
        "NH3_working_capacity_mol_kg": working_capacity_mol_kg,
        "NH3_working_capacity_mmol_g": working_capacity_mmol_g,
        "NH3_working_capacity_mg_g": working_capacity_mg_g,
        "NH3_heat_of_adsorption_adsorption_kcal_mol": heat_of_adsorption_NH3_ads,
        "NH3_heat_of_adsorption_desorption_kcal_mol": heat_of_adsorption_NH3_des,
    }
    
    return output


def run_nh3_simulation_or_fail(cif_path, **kwargs):
    """Run NH3 uptake simulation with error handling."""
    try:
        return nh3_uptake_simulation(cif_path, **kwargs)
    except Exception as e:
        print(e)
        return None
