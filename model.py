import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Region:
    """Represents a geographical region for transplant modeling"""
    name: str
    population: int
    hospitals: int
    cold_chain_capacity: int
    surgeon_teams: int
    gdp_per_capita: float

@dataclass
class Costs:
    """Cost structure for xenotransplantation"""
    facility_construction: float  # per facility
    facility_operation: float    # per year
    organ_processing: float     # per organ
    transplant_procedure: float # per procedure
    training_per_team: float    # per surgical team
    cold_chain_per_mile: float  # per mile of transport

class SurvivalModel:
    """More sophisticated survival analysis"""
    def __init__(self):
        # Weibull parameters for different scenarios
        self.survival_params = {
            'waitlist': {'shape': 1.5, 'scale': 5},
            'post_xeno': {'shape': 1.8, 'scale': 8},
            'post_human': {'shape': 2.0, 'scale': 10}
        }
        
        # Quality of life multipliers (0-1)
        self.qol_multipliers = {
            'waitlist': 0.5,
            'post_xeno': 0.8,
            'post_human': 0.9
        }
    
    def calculate_survival_probability(self, time: float, scenario: str) -> float:
        """Calculate survival probability for given time and scenario"""
        params = self.survival_params[scenario]
        return weibull_min.sf(time, params['shape'], scale=params['scale'])
    
    def calculate_qalys(self, years: float, scenario: str) -> float:
        """Calculate Quality Adjusted Life Years"""
        survival_prob = self.calculate_survival_probability(years, scenario)
        return survival_prob * self.qol_multipliers[scenario] * years

class RegionalDistribution:
    """Handles regional aspects of xenotransplantation"""
    def __init__(self):
        # Example regions
        self.regions: Dict[str, Region] = {
            'northeast': Region('Northeast', 55000000, 200, 1000, 50, 70000),
            'southeast': Region('Southeast', 65000000, 180, 900, 45, 55000),
            'midwest': Region('Midwest', 45000000, 150, 800, 40, 58000),
            'southwest': Region('Southwest', 40000000, 120, 600, 35, 56000),
            'west': Region('West', 50000000, 160, 700, 42, 72000)
        }
        
        # Transport network model (distances between regions)
        self.distances = pd.DataFrame({
            'from_region': [],
            'to_region': [],
            'distance': []
        })
        
    def calculate_regional_demand(self, organ_type: str) -> pd.DataFrame:
        """Calculate demand by region based on population and demographics"""
        regional_demand = []
        for region in self.regions.values():
            # Simplified demand calculation
            base_demand = region.population * 0.0003  # example rate
            adjusted_demand = base_demand * (50000 / region.gdp_per_capita)
            
            regional_demand.append({
                'region': region.name,
                'demand': adjusted_demand,
                'capacity': region.hospitals * 50  # example capacity calculation
            })
        return pd.DataFrame(regional_demand)

class FacilityModel:
    """Enhanced facility scaling and operations model"""
    def __init__(self, costs: Costs,
                 initial_annual_capacity: int,
                 mature_annual_capacity: int,
                 capacity_growth_rate: float,
                 initial_surgical_teams: int,
                 surgical_team_growth_rate: float,
                 max_surgical_teams: int):
        
        # Current number of transplant teams by organ type (2023 data)
        self.current_teams = {
            'kidney': 250,  # ~250 kidney transplant centers
            'heart': 150,   # ~150 heart transplant centers
            'liver': 150,   # ~150 liver transplant centers
            'lung': 75,     # ~75 lung transplant centers
            'pancreas': 50  # ~50 pancreas transplant centers
        }
        
        self.initial_capacity = initial_annual_capacity
        self.mature_capacity = mature_annual_capacity
        self.capacity_growth_rate = capacity_growth_rate
        
        # Surgical team constraints
        self.initial_surgical_teams = initial_surgical_teams
        self.surgical_team_growth_rate = surgical_team_growth_rate
        self.max_teams_multiplier = 3  # Can expand current capacity by 3x
        
        self.costs = costs
        self.learning_rate = 0.85
    
    def calculate_unit_cost(self, cumulative_production: float) -> float:
        """Calculate unit cost based on learning curve effects"""
        if cumulative_production == 0:
            return self.costs.organ_processing
        
        # Apply learning curve
        cost_reduction = (cumulative_production / 1000) ** (np.log2(self.learning_rate))
        return max(
            self.costs.organ_processing * cost_reduction,
            self.costs.organ_processing * 0.3  # Floor at 30% of initial cost
        )
    
    def calculate_annual_capacity(self, year: int) -> float:
        """Calculate annual production capacity based on growth"""
        capacity = self.initial_capacity * (1 + self.capacity_growth_rate)**year
        return min(capacity, self.mature_capacity)
    
    def calculate_available_surgeries(self, year: int, organ_type: str) -> int:
        """Calculate number of possible surgeries based on team constraints"""
        max_teams = self.current_teams[organ_type] * self.max_teams_multiplier
        teams = self.initial_surgical_teams * (1 + self.surgical_team_growth_rate)**year
        teams = min(teams, max_teams)
        return int(teams * 24)  # Each team can do 24 surgeries per year

class EnhancedXenoTransplantScaling:
    """Enhanced version of original XenoTransplantScaling"""
    def __init__(self, costs: Costs, regions: RegionalDistribution, facility_model: FacilityModel, growth_rates: Dict[str, float] = None):
        self.costs = costs
        self.regions = regions
        self.facility_model = facility_model
        self.survival_model = SurvivalModel()
        
        # Use provided growth rates or defaults
        self.traditional_growth_rates = growth_rates or {
            'deceased': 0.01,
            'living': 0.005
        }
        
        # Add these new attributes
        self.potential_recipient_multipliers = {
            'kidney': 3.0,  # 3x current waitlist (many ESRD patients never listed)
            'heart': 2.5,   # 2.5x (NYHA Class IV patients not listed)
            'liver': 2.0,   # 2x (MELD scores too low)
            'lung': 2.0,    # 2x (too sick to list)
            'pancreas': 1.5 # 1.5x (less restrictive criteria)
        }
        
        self.alternative_treatment_costs = {
            'kidney': {
                'treatment': 'Dialysis',
                'annual_cost': 90000,  # Annual cost of dialysis
                'qaly_score': 0.6      # Quality of life score
            },
            'heart': {
                'treatment': 'LVAD',
                'annual_cost': 150000,  # Annual LVAD maintenance
                'qaly_score': 0.5
            },
            'liver': {
                'treatment': 'Medical Management',
                'annual_cost': 70000,
                'qaly_score': 0.4
            },
            'lung': {
                'treatment': 'Oxygen Therapy',
                'annual_cost': 50000,
                'qaly_score': 0.5
            },
            'pancreas': {
                'treatment': 'Insulin Therapy',
                'annual_cost': 20000,
                'qaly_score': 0.7
            }
        }
    
    def project_organ_supply(self, years: int, organ_type: str, scenario: str) -> pd.DataFrame:
        """Project organ supply over the specified number of years"""
        print("\n=== Starting project_organ_supply ===")
        results = []
        
        # Calculate facilities based on scenario
        new_facilities_per_year = {
            'conservative': 0.25,  # One new facility every 4 years
            'moderate': 0.5,      # One new facility every 2 years
            'aggressive': 1.0     # One new facility per year
        }
        
        current_facilities = 1.0  # Start with one facility
        
        for year in range(years):
            # Calculate production capacity PER FACILITY
            per_facility_capacity = self.facility_model.calculate_annual_capacity(year)
            
            # Total capacity across all facilities
            total_capacity = per_facility_capacity * current_facilities
            
            # Calculate surgical constraint based on organ type
            available_surgeries = self.facility_model.calculate_available_surgeries(year, organ_type)
            
            # Take the minimum of capacity and surgical constraint
            organ_supply = min(total_capacity, available_surgeries)
            
            results.append({
                'year': year,
                'facilities': current_facilities,
                'per_facility_capacity': per_facility_capacity,
                'total_capacity': total_capacity,
                'available_surgeries': available_surgeries,
                'organ_supply': organ_supply
            })
            
            # Update facilities for next year
            current_facilities += new_facilities_per_year[scenario]
        
        return pd.DataFrame(results)
    
    def project_scaled_implementation(self, years: int, organ_type: str, scenario: str) -> pd.DataFrame:
        """Project complete scaled implementation including regional distribution"""
        base_supply = self.project_organ_supply(years, organ_type, scenario)
        regional_demand = self.regions.calculate_regional_demand(organ_type)
        
        results = []
        cumulative_production = 0
        cumulative_investment = 0
        
        for year in range(years):
            year_supply = float(base_supply.iloc[year]['organ_supply'])
            facility_count = float(base_supply.iloc[year]['facilities'])
            
            # Calculate facility costs
            facility_costs = facility_count * self.costs.facility_construction / 10  # Amortized over 10 years
            operational_costs = facility_count * self.costs.facility_operation
            
            # Calculate unit cost and processing costs
            unit_cost = self.facility_model.calculate_unit_cost(cumulative_production)
            processing_costs = year_supply * unit_cost
            
            total_costs = facility_costs + operational_costs + processing_costs
            cumulative_investment += total_costs
            
            results.append({
                'year': year,
                'total_supply': year_supply,
                'cumulative_production': cumulative_production,
                'facility_costs': facility_costs,
                'operational_costs': operational_costs,
                'processing_costs': processing_costs,
                'total_costs': total_costs,
                'cumulative_investment': cumulative_investment,
                'unit_cost': unit_cost,
                'qalys_gained': year_supply * (
                    self.survival_model.calculate_qalys(1, 'post_xeno') -
                    self.survival_model.calculate_qalys(1, 'waitlist')
                ),
                'regional_distribution': self._distribute_supply(year_supply, regional_demand)
            })
            
            cumulative_production += year_supply
        
        return pd.DataFrame(results)
    
    def _distribute_supply(self, total_supply: float, 
                         regional_demand: pd.DataFrame) -> Dict[str, float]:
        """Distribute available supply across regions based on demand"""
        total_demand = regional_demand['demand'].sum()
        distribution = {}
        
        for _, region in regional_demand.iterrows():
            share = region['demand'] / total_demand
            allocated = min(total_supply * share, region['capacity'])
            distribution[region['region']] = allocated
            
        return distribution
    
    def project_all_transplant_types(self, years: int, organ_type: str, scenario: str) -> pd.DataFrame:
        """Project all types of transplants including xeno, deceased, and living donors"""
        print("\n=== Starting project_all_transplant_types ===")
        print(f"Growth rates being used: deceased={self.traditional_growth_rates['deceased']}, living={self.traditional_growth_rates['living']}")
        
        base_supply = self.project_organ_supply(years, organ_type, scenario)
        
        # Base rates for traditional transplants
        base_rates = {
            'kidney': {'deceased': 15000, 'living': 6000},
            'heart': {'deceased': 3500, 'living': 0},
            'liver': {'deceased': 8000, 'living': 500},
            'lung': {'deceased': 2500, 'living': 0},
            'pancreas': {'deceased': 1000, 'living': 0}
        }
        
        print(f"\nBase rates for {organ_type}:")
        print(f"- Deceased: {base_rates[organ_type]['deceased']}")
        print(f"- Living: {base_rates[organ_type]['living']}")
        
        results = []
        for year in range(years):
            deceased = base_rates[organ_type]['deceased'] * (1 + self.traditional_growth_rates['deceased'])**year
            living = base_rates[organ_type]['living'] * (1 + self.traditional_growth_rates['living'])**year
            xeno = float(base_supply.iloc[year]['organ_supply'])
            
            if year % 5 == 0:  # Log every 5 years to avoid too much output
                print(f"\nYear {year} calculations:")
                print(f"- Deceased: {deceased:.0f}")
                print(f"- Living: {living:.0f}")
                print(f"- Xeno: {xeno:.0f}")
                print(f"- Total: {(deceased + living + xeno):.0f}")
            
            results.append({
                'year': year,
                'Xenotransplants': xeno,
                'Deceased Donor': deceased,
                'Living Donor': living,
                'Total': xeno + deceased + living
            })
        
        print("\n=== Final Results ===")
        df = pd.DataFrame(results)
        print(df)
        return df
    
    def calculate_waitlist_deaths(self, years: int, organ_type: str, scenario: str) -> pd.DataFrame:
        """Calculate expected deaths on waitlist with and without xenotransplantation"""
        
        # Add mortality rates by organ type (annual rates)
        mortality_rates = {
            'kidney': 0.067,  # 6.7% annual mortality on kidney waitlist
            'heart': 0.17,    # 17% annual mortality on heart waitlist
            'liver': 0.12,    # 12% annual mortality on liver waitlist
            'lung': 0.15,     # 15% annual mortality on lung waitlist
            'pancreas': 0.05  # 5% annual mortality on pancreas waitlist
        }
        
        # More conservative growth rates
        waitlist_growth_rates = {
            'kidney': 0.015,  # 1.5% annual growth
            'heart': 0.02,    # 2% annual growth
            'liver': 0.015,   # 1.5% annual growth
            'lung': 0.02,     # 2% annual growth
            'pancreas': 0.01  # 1% annual growth
        }
        
        # Initial conditions (based on OPTN data 2023)
        initial_conditions = {
            'kidney': {
                'waitlist_size': 90000,
                'annual_additions': 30000,
                'annual_transplants': 25000,
            },
            'heart': {
                'waitlist_size': 3500,
                'annual_additions': 4000,
                'annual_transplants': 3500,
            },
            'liver': {
                'waitlist_size': 12000,
                'annual_additions': 12000,
                'annual_transplants': 9000,
            },
            'lung': {
                'waitlist_size': 1000,
                'annual_additions': 3000,
                'annual_transplants': 2500,
            },
            'pancreas': {
                'waitlist_size': 1200,
                'annual_additions': 1500,
                'annual_transplants': 1000,
            }
        }
        
        # Get initial values for this organ type
        init = initial_conditions[organ_type]
        mortality_rate = mortality_rates[organ_type]
        growth_rate = waitlist_growth_rates[organ_type]
        
        # Get transplant projections including xenotransplantation
        transplant_data = self.project_all_transplant_types(years, organ_type, scenario)
        
        results = []
        
        # Initialize tracking variables (separate for each scenario)
        baseline_waitlist = init['waitlist_size']
        xeno_waitlist = init['waitlist_size']
        annual_additions = init['annual_additions']
        
        for year in range(years):
            # Update annual additions with growth
            annual_additions *= (1 + growth_rate)
            
            # BASELINE SCENARIO (without xenotransplantation)
            baseline_transplants = (transplant_data.loc[year, 'Deceased Donor'] + 
                                  transplant_data.loc[year, 'Living Donor'])
            
            # Update baseline waitlist
            baseline_waitlist = (baseline_waitlist +      # Previous waitlist
                                 annual_additions -     # New additions
                                 baseline_transplants)  # Transplants performed
            
            # Calculate deaths for baseline scenario
            baseline_deaths = baseline_waitlist * mortality_rate
            baseline_waitlist -= baseline_deaths  # Remove deaths from waitlist
            
            # XENOTRANSPLANTATION SCENARIO
            xeno_transplants = transplant_data.loc[year, 'Total']
            
            # Update xeno waitlist
            xeno_waitlist = (xeno_waitlist +      # Previous waitlist
                             annual_additions -     # New additions
                             xeno_transplants)      # All transplants including xeno
            
            # Calculate deaths for xeno scenario
            xeno_deaths = max(0, xeno_waitlist * mortality_rate)
            xeno_waitlist -= xeno_deaths  # Remove deaths from waitlist
            
            results.append({
                'year': year,
                'baseline_deaths': baseline_deaths,
                'deaths_with_xeno': xeno_deaths,
                'lives_saved': baseline_deaths - xeno_deaths,
                'baseline_waitlist': baseline_waitlist,
                'waitlist_with_xeno': xeno_waitlist,
                'annual_additions': annual_additions,
                'baseline_transplants': baseline_transplants,
                'xeno_transplants': xeno_transplants
            })
        
        return pd.DataFrame(results)
    
    def calculate_expanded_access_impact(self, years: int, organ_type: str, 
                                  scenario: str) -> pd.DataFrame:
        """Calculate impact of expanded access with xenotransplantation"""
        
        base_results = self.calculate_waitlist_deaths(years, organ_type, scenario)
        
        # Get potential recipient multiplier
        multiplier = self.potential_recipient_multipliers[organ_type]
        alt_treatment = self.alternative_treatment_costs[organ_type]
        
        # Calculate expanded population
        expanded_results = []
        cumulative_production = 0
        
        for year in range(years):
            current_row = base_results.iloc[year]
            
            # Calculate potential recipients who could benefit
            potential_recipients = current_row['baseline_waitlist'] * (multiplier - 1)
            
            # Calculate how many could be served with xeno supply
            available_organs = current_row['xeno_transplants']
            additional_served = min(potential_recipients, 
                                  max(0, available_organs - current_row['baseline_waitlist']))
            
            # Calculate costs using facility model
            unit_cost = self.facility_model.calculate_unit_cost(cumulative_production)
            xenotransplant_cost = unit_cost * additional_served
            alternative_cost = alt_treatment['annual_cost'] * potential_recipients
            
            expanded_results.append({
                'year': year,
                'additional_recipients': potential_recipients,
                'additional_served': additional_served,
                'alternative_treatment_cost': alternative_cost,
                'xenotransplant_cost': xenotransplant_cost,
                'qaly_gained': additional_served * (
                    self.survival_model.calculate_qalys(1, 'post_xeno') - 
                    alt_treatment['qaly_score']
                )
            })
            
            cumulative_production += additional_served
        
        return pd.DataFrame(expanded_results)

def run_enhanced_analysis(
    organ_type: str, 
    years: int = 10, 
    scenario: str = 'conservative',
    growth_rates: Dict[str, float] = None,
    initial_capacity: int = 100,
    mature_capacity: int = 500,
    capacity_growth_rate: float = 0.15,
    initial_surgical_teams: int = 5,
    surgical_team_growth_rate: float = 0.2,
    max_surgical_teams: int = 50,
    surgeries_per_team: int = 24
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
    """Run complete analysis with all enhancements"""
    
    # Initialize costs
    costs = Costs(
        facility_construction=100_000_000,
        facility_operation=20_000_000,
        organ_processing=50_000,
        transplant_procedure=250_000,
        training_per_team=500_000,
        cold_chain_per_mile=100
    )
    
    # Initialize models with parameters
    facility_model = FacilityModel(
        costs=costs,
        initial_annual_capacity=initial_capacity,
        mature_annual_capacity=mature_capacity,
        capacity_growth_rate=capacity_growth_rate,
        initial_surgical_teams=initial_surgical_teams,
        surgical_team_growth_rate=surgical_team_growth_rate,
        max_surgical_teams=max_surgical_teams
    )
    
    regions = RegionalDistribution()
    
    # Create scaling model
    scaling_model = EnhancedXenoTransplantScaling(
        costs=costs,
        regions=regions,
        facility_model=facility_model,
        growth_rates=growth_rates or {'deceased': 0.01, 'living': 0.005}
    )
    
    # Get projections using the model with updated growth rates
    implementation_proj = scaling_model.project_scaled_implementation(years, organ_type, scenario)
    all_transplants = scaling_model.project_all_transplant_types(years, organ_type, scenario)
    waitlist_deaths = scaling_model.calculate_waitlist_deaths(years, organ_type, scenario)
    
    # Calculate financial metrics
    total_costs = implementation_proj['total_costs'].sum()
    total_qalys = implementation_proj['qalys_gained'].sum()
    cost_per_qaly = total_costs / total_qalys
    
    summary_metrics = {
        'total_investment_required': total_costs,
        'total_qalys_gained': total_qalys,
        'cost_per_qaly': cost_per_qaly,
        'cumulative_organs_produced': implementation_proj['cumulative_production'].max(),
        'final_unit_cost': implementation_proj['unit_cost'].iloc[-1]
    }
    
    return implementation_proj, summary_metrics, all_transplants, waitlist_deaths

# Example usage:
if __name__ == "__main__":
    projections, metrics, all_transplants, waitlist_deaths = run_enhanced_analysis('kidney', years=10, scenario='conservative')
    print("\nProjections:")
    print(projections)
    print("\nSummary Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: ${value:,.2f}")