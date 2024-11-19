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
    def __init__(self, costs: Costs):
        self.costs = costs
        self.learning_rate = 0.9  # 10% cost reduction per doubling of output
        self.startup_time = 180  # days from construction to first organ
        self.max_capacity_utilization = 0.85
        
        # Facility development stages
        self.stages = {
            'planning': 90,  # days
            'construction': 360,
            'certification': 90,
            'initial_operation': 180
        }
    
    def calculate_facility_ramp_up(self, days_operating: int) -> float:
        """Calculate capacity utilization based on facility age"""
        if days_operating < self.startup_time:
            return 0.0
        ramp_up = min((days_operating - self.startup_time) / 360, 1.0)
        return ramp_up * self.max_capacity_utilization
    
    def calculate_unit_cost(self, cumulative_production: int) -> float:
        """Calculate unit cost with learning curve effects"""
        if cumulative_production <= 0:
            return self.costs.organ_processing
        # Use learning curve formula: Cost_n = Cost_1 * n^(log_2(learning_rate))
        return self.costs.organ_processing * (cumulative_production ** (np.log2(self.learning_rate)))

class EnhancedXenoTransplantScaling:
    """Enhanced version of original XenoTransplantScaling"""
    def __init__(self, costs: Costs, regions: RegionalDistribution):
        self.costs = costs
        self.regions = regions
        self.facility_model = FacilityModel(costs)
        self.survival_model = SurvivalModel()
        
    def project_organ_supply(self, years: int, organ_type: str, scenario: str) -> pd.DataFrame:
        """Project organ supply over the specified number of years"""
        results = []
        
        # Base assumptions
        initial_facilities = 2
        facility_capacity = 1000  # organs per year
        growth_rates = {
            'conservative': 1.2,  # 20% year-over-year growth
            'moderate': 1.4,     # 40% year-over-year growth
            'aggressive': 1.6    # 60% year-over-year growth
        }
        
        facilities = initial_facilities
        for year in range(years):
            # Calculate supply based on facilities and utilization
            utilization = self.facility_model.calculate_facility_ramp_up(year * 365)
            organ_supply = facilities * facility_capacity * utilization
            
            results.append({
                'year': year,
                'facilities': facilities,
                'organ_supply': organ_supply
            })
            
            # Grow facilities according to scenario
            facilities *= growth_rates[scenario]
            
        return pd.DataFrame(results)
    
    def project_scaled_implementation(self, years: int, organ_type: str, 
                                   scenario: str = 'conservative') -> pd.DataFrame:
        """Project complete scaled implementation including regional distribution"""
        base_supply = self.project_organ_supply(years, organ_type, scenario)
        regional_demand = self.regions.calculate_regional_demand(organ_type)
        
        results = []
        cumulative_production = 0
        
        for year in range(years):
            year_supply = float(base_supply.iloc[year]['organ_supply'])
            facility_count = float(base_supply.iloc[year]['facilities'])
            
            # Calculate costs with safety checks
            facility_costs = facility_count * (self.costs.facility_construction / 10 + self.costs.facility_operation)
            
            # Calculate unit cost and processing costs
            unit_cost = float(self.facility_model.calculate_unit_cost(cumulative_production))
            processing_costs = float(year_supply * unit_cost) if year_supply > 0 else 0.0
            
            total_costs = float(facility_costs + processing_costs)
            
            # Calculate QALYs
            qalys_gained = float(year_supply * 
                           (self.survival_model.calculate_qalys(1, 'post_xeno') -
                            self.survival_model.calculate_qalys(1, 'waitlist')))
            
            # Distribute supply across regions
            regional_distribution = self._distribute_supply(year_supply, regional_demand)
            
            results.append({
                'year': year,
                'total_supply': year_supply,
                'cumulative_production': cumulative_production,
                'total_costs': total_costs,
                'unit_cost': unit_cost,
                'qalys_gained': qalys_gained,
                'regional_distribution': regional_distribution
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

def run_enhanced_analysis(organ_type: str, years: int = 10, 
                         scenario: str = 'conservative') -> Tuple[pd.DataFrame, Dict]:
    """Run complete analysis with all enhancements"""
    
    # Initialize costs
    costs = Costs(
        facility_construction=100_000_000,  # $100M per facility
        facility_operation=20_000_000,      # $20M per year
        organ_processing=50_000,            # $50K per organ
        transplant_procedure=250_000,       # $250K per procedure
        training_per_team=500_000,          # $500K per team
        cold_chain_per_mile=100            # $100 per mile
    )
    
    # Initialize models
    regions = RegionalDistribution()
    # demand_model = TransplantDemandModel()
    scaling_model = EnhancedXenoTransplantScaling(costs, regions)
    
    # demand_proj = demand_model.project_waitlist(years, organ_type)
    implementation_proj = scaling_model.project_scaled_implementation(years, organ_type, scenario)
    
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
    
    return implementation_proj, summary_metrics

# Example usage:
if __name__ == "__main__":
    projections, metrics = run_enhanced_analysis('kidney', years=10, scenario='conservative')
    print("\nProjections:")
    print(projections)
    print("\nSummary Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: ${value:,.2f}")