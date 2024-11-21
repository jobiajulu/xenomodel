# At the top of the file, ensure these are the exported names
__all__ = [
    'Costs',
    'RegionalDistribution',
    'EnhancedFacilityModel',
    'EnhancedXenoTransplantScaling',
    'run_enhanced_analysis'
]

import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import plotly.graph_objects as go

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
    """Enhanced cost structure for xenotransplantation"""
    # Facility costs
    facility_construction: float  # In USD (millions * 1_000_000)
    facility_operation: float    # In USD (millions * 1_000_000)
    
    # Per-organ costs
    organ_processing: float     # In USD
    transplant_procedure: float # In USD
    
    # Support costs
    training_per_team: float    # In USD
    cold_chain_per_mile: float  # In USD
    
    # Alternative treatment costs
    alternative_costs: Dict[str, Dict] = field(default_factory=lambda: {
        'kidney': {'annual_cost': 90000, 'qaly_score': 0.6},  # Dialysis
        'heart': {'annual_cost': 150000, 'qaly_score': 0.5},  # LVAD
        'liver': {'annual_cost': 70000, 'qaly_score': 0.4},   # Medical Management
        'lung': {'annual_cost': 50000, 'qaly_score': 0.5},    # Oxygen Therapy
        'pancreas': {'annual_cost': 20000, 'qaly_score': 0.7} # Insulin Therapy
    })
    
    # Phase-specific multipliers
    phase_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'experimental': 2.5,    # Higher costs during trials
        'scaling': 1.5,        # Moderate premium during scaling
        'mature': 1.0          # Base costs when mature
    })

@dataclass
class Phase:
    """Represents a development phase with its characteristics"""
    duration: int
    success_rate: float
    regulatory_costs: float
    training_requirements: float
    monitoring_costs: float
    quality_control_factor: float

class CostTracker:
    """Enhanced cost tracking with reporting capabilities"""
    def __init__(self):
        self.cost_history = []  # List of dictionaries with cost entries
        self.phase_transitions = []
        self.cost_milestones = []
        
        # Initialize empty phase costs
        self.phase_costs = {
            'experimental': defaultdict(float),
            'scaling': defaultdict(float),
            'mature': defaultdict(float)
        }
    
    def record_cost(self, phase: str, category: str, amount: float, year: int):
        """Record a cost entry with metadata"""
        if amount <= 0:
            return
            
        entry = {
            'year': year,
            'phase': phase,  # Ensure phase is included
            'category': category,
            'amount': amount
        }
        
        self.cost_history.append(entry)
        self.phase_costs[phase][category] += amount
    
    def generate_visualizations(self) -> Dict[str, go.Figure]:
        """Generate plotly figures for cost analysis"""
        if not self.cost_history:  # Check if we have data
            return self._generate_empty_visualizations()
            
        return {
            'phase_costs': self._create_phase_cost_figure(),
            'cost_breakdown': self._create_cost_breakdown_figure(),
            'transition_analysis': self._create_transition_figure()
        }
    
    def _generate_empty_visualizations(self) -> Dict[str, go.Figure]:
        """Generate empty placeholder visualizations"""
        def create_empty_figure(title: str) -> go.Figure:
            fig = go.Figure()
            fig.add_annotation(
                text="No cost data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            fig.update_layout(
                title=title,
                xaxis_title="",
                yaxis_title="",
                showlegend=False
            )
            return fig
        
        return {
            'phase_costs': create_empty_figure("Phase Cost Analysis"),
            'cost_breakdown': create_empty_figure("Cost Breakdown Over Time"),
            'transition_analysis': create_empty_figure("Phase Transitions")
        }
    
    def _create_phase_cost_figure(self) -> go.Figure:
        """Create phase cost comparison visualization"""
        if not self.cost_history:
            return self._generate_empty_visualizations()['phase_costs']
            
        df = pd.DataFrame(self.cost_history)
        phase_totals = df.groupby(['phase', 'category'])['amount'].sum().reset_index()
        
        fig = go.Figure()
        for phase in ['experimental', 'scaling', 'mature']:
            phase_data = phase_totals[phase_totals['phase'] == phase]
            if not phase_data.empty:
                fig.add_trace(go.Bar(
                    name=phase,
                    x=phase_data['category'],
                    y=phase_data['amount'],
                    text=phase_data['amount'].apply(lambda x: f'${x:,.0f}'),
                    textposition='auto',
                ))
        
        fig.update_layout(
            title='Cost Distribution by Phase',
            barmode='group',
            xaxis_title='Cost Category',
            yaxis_title='Total Cost (USD)',
            showlegend=True
        )
        return fig
    
    def _generate_cost_trajectory(self) -> pd.DataFrame:
        """Generate cost trajectory over time"""
        return pd.DataFrame(self.cost_history)
    
    def _generate_phase_summary(self) -> pd.DataFrame:
        """Generate summary of costs by phase"""
        phase_data = []
        for phase in ['experimental', 'scaling', 'mature']:
            phase_costs = {
                'phase': phase,
                'total_cost': sum(item['amount'] for item in self.cost_history 
                                if item['phase'] == phase),
                'duration_years': len(set(item['year'] for item in self.cost_history 
                                       if item['phase'] == phase)),
                'categories': len(set(item['category'] for item in self.cost_history 
                                    if item['phase'] == phase))
            }
            phase_data.append(phase_costs)
        return pd.DataFrame(phase_data)
    
    def _create_cost_breakdown_figure(self) -> go.Figure:
        """Create detailed cost breakdown visualization"""
        df = pd.DataFrame(self.cost_history)
        yearly_costs = df.pivot_table(
            index='year',
            columns='category',
            values='amount',
            aggfunc='sum'
        ).fillna(0)
        
        fig = go.Figure()
        for category in yearly_costs.columns:
            fig.add_trace(go.Scatter(
                name=category,
                x=yearly_costs.index,
                y=yearly_costs[category],
                stackgroup='one',
                hovertemplate="%{y:$,.0f}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Cumulative Cost Breakdown Over Time',
            xaxis_title='Year',
            yaxis_title='Cost (USD)',
            showlegend=True,
            hovermode='x unified'
        )
        return fig
    
    def _create_transition_figure(self) -> go.Figure:
        """Create phase transition analysis visualization"""
        transitions = pd.DataFrame(self.phase_transitions)
        
        fig = go.Figure()
        
        # Add phase bands
        for i, row in transitions.iterrows():
            fig.add_vrect(
                x0=row['year'],
                x1=transitions.iloc[i+1]['year'] if i < len(transitions)-1 else transitions['year'].max(),
                fillcolor={"experimental": "rgba(255,0,0,0.1)",
                          "scaling": "rgba(0,255,0,0.1)",
                          "mature": "rgba(0,0,255,0.1)"}[row['to_phase']],
                layer="below",
                line_width=0,
                annotation_text=row['to_phase'],
                annotation_position="top left"
            )
        
        # Add cost trajectory
        df = pd.DataFrame(self.cost_history)
        yearly_total = df.groupby('year')['amount'].sum().cumsum()
        
        fig.add_trace(go.Scatter(
            x=yearly_total.index,
            y=yearly_total.values,
            name='Cumulative Cost',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title='Phase Transitions and Cumulative Cost',
            xaxis_title='Year',
            yaxis_title='Cumulative Cost (USD)',
            showlegend=True
        )
        return fig

class SurvivalModel:
    """Enhanced survival analysis with equity considerations"""
    def __init__(self):
        # Updated Weibull parameters based on latest data
        self.survival_params = {
            'waitlist': {'shape': 1.5, 'scale': 5},
            'post_xeno': {'shape': 1.8, 'scale': 8},
            'post_human': {'shape': 2.0, 'scale': 10},
            'alternative_treatment': {'shape': 1.3, 'scale': 4}
        }
        
        # Updated QoL multipliers with equity considerations
        self.qol_multipliers = {
            'waitlist': 0.5,
            'post_xeno': 0.8,
            'post_human': 0.9,
            'alternative_treatment': {
                'kidney': 0.6,  # Dialysis
                'heart': 0.5,   # LVAD
                'liver': 0.4,   # Medical management
                'lung': 0.5,    # Oxygen therapy
                'pancreas': 0.7 # Insulin therapy
            }
        }
        
    def calculate_survival_probability(self, time: float, scenario: str) -> float:
        """Calculate survival probability for given time and scenario"""
        params = self.survival_params[scenario]
        return weibull_min.sf(time, params['shape'], scale=params['scale'])
    
    def calculate_qalys(self, years: float, scenario: str) -> float:
        """Calculate Quality Adjusted Life Years"""
        survival_prob = self.calculate_survival_probability(years, scenario)
        return survival_prob * self.qol_multipliers[scenario] * years
    
    def calculate_equity_adjusted_qalys(self, 
                                      years: float, 
                                      scenario: str,
                                      demographic_factors: Dict[str, float]) -> float:
        """Calculate QALYs with equity adjustments"""
        base_qalys = self.calculate_qalys(years, scenario)
        equity_multiplier = self.calculate_equity_multiplier(demographic_factors)
        return base_qalys * equity_multiplier

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

class OrganDemandData:
    """Centralized data for organ-specific demand and outcomes"""
    ORGAN_DEMAND = {
        'kidney': {
            'waitlist_size': 90000,
            'annual_deaths': 12000,
            'total_treatment_baseline': 250000,
            'growth_rate': 0.04,
            'annual_esrd_deaths': 50000,
            'esrd_mortality_rate': 0.20,
            'deceased_annual': 15000,  # Annual deceased donor transplants
            'living_annual': 6000      # Annual living donor transplants
        },
        'heart': {
            'waitlist_size': 3500,
            'annual_deaths': 1000,
            'total_treatment_baseline': 50000,
            'growth_rate': 0.03,
            'annual_eshd_deaths': 25000,
            'eshd_mortality_rate': 0.50,
            'deceased_annual': 3500,    # Annual deceased donor transplants
            'living_annual': 0          # No living donors for hearts
        },
        'liver': {
            'waitlist_size': 12000,
            'annual_deaths': 2500,
            'total_treatment_baseline': 40000,
            'growth_rate': 0.03,
            'annual_esld_deaths': 20000,
            'esld_mortality_rate': 0.40,
            'deceased_annual': 8000,    # Annual deceased donor transplants
            'living_annual': 500        # Annual living donor transplants
        },
        'lung': {
            'waitlist_size': 1000,
            'annual_deaths': 400,
            'total_treatment_baseline': 15000,
            'growth_rate': 0.02,
            'annual_espd_deaths': 10000,
            'espd_mortality_rate': 0.45,
            'deceased_annual': 2500,    # Annual deceased donor transplants
            'living_annual': 0          # No living donors for lungs
        },
        'pancreas': {
            'waitlist_size': 1200,
            'annual_deaths': 200,
            'total_treatment_baseline': 8000,
            'growth_rate': 0.02,
            'annual_espad_deaths': 3000,
            'espad_mortality_rate': 0.15,
            'deceased_annual': 800,     # Annual deceased donor transplants
            'living_annual': 0          # No living donors for pancreas
        }
    }

    @classmethod
    def validate_data(cls):
        """Validate that all required fields are present in the data"""
        required_fields = [
            'waitlist_size', 'annual_deaths', 'total_treatment_baseline', 
            'growth_rate', 'annual_esrd_deaths', 'esrd_mortality_rate'
        ]
        
        for organ, data in cls.ORGAN_DEMAND.items():
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                raise ValueError(f"Missing required fields for {organ}: {missing_fields}")

class EnhancedFacilityModel:
    """Enhanced facility model with detailed phase tracking"""
    def __init__(self, costs: Costs):
        self.costs = costs
        self.cost_tracker = CostTracker()
        self.current_phase = 'experimental'
        self.last_transition_year = 0
        
        # Define development phases with consistent naming
        self.phases = {
            'experimental': Phase(
                duration=5,
                success_rate=0.60,
                regulatory_costs=10_000_000,  # $10M annually
                training_requirements=2_000_000,  # $2M annually
                monitoring_costs=5_000_000,  # $5M annually
                quality_control_factor=2.0  # 2x standard QC costs
            ),
            'scaling': Phase(  # Changed from 'early_clinical' to 'scaling'
                duration=5,
                success_rate=0.75,
                regulatory_costs=5_000_000,  # $5M annually
                training_requirements=1_000_000,  # $1M annually
                monitoring_costs=2_500_000,  # $2.5M annually
                quality_control_factor=1.5  # 1.5x standard QC costs
            ),
            'mature': Phase(  # Changed from 'established' to 'mature'
                duration=float('inf'),  # Continues indefinitely
                success_rate=0.90,
                regulatory_costs=2_000_000,  # $2M annually
                training_requirements=500_000,  # $500k annually
                monitoring_costs=1_000_000,  # $1M annually
                quality_control_factor=1.0  # Standard QC costs
            )
        }
        
        # Initialize other parameters
        self.capacity_params = {
            'initial': 100,
            'mature': 500,
            'growth_rate': 0.15
        }
        
        self.surgical_params = {
            'initial_teams': 5,
            'growth_rate': 0.20,
            'surgeries_per_team': 24,
            'max_teams': float('inf')
        }
        
        # Treatment need data with sources
        self.organ_demand = OrganDemandData.ORGAN_DEMAND
    
    def calculate_annual_capacity(self, year: int) -> float:
        """Calculate annual production capacity"""
        initial = self.capacity_params['initial']
        mature = self.capacity_params['mature']
        growth_rate = self.capacity_params['growth_rate']
        
        # Logistic growth function for capacity
        capacity = initial + (mature - initial) * (
            1 / (1 + np.exp(-growth_rate * (year - 5)))
        )
        
        # Apply phase-specific success rates
        current_phase = self.determine_phase(year)
        success_rate = self.phases[current_phase].success_rate
        
        return capacity * success_rate
    
    def calculate_available_surgeries(self, year: int, organ_type: str) -> float:
        """Calculate available surgical capacity"""
        params = self.surgical_params
        
        # Calculate number of surgical teams
        current_teams = params['initial_teams'] * (
            1 + params['growth_rate']
        ) ** year
        current_teams = min(current_teams, params['max_teams'])
        
        # Calculate total available surgeries
        available_surgeries = current_teams * params['surgeries_per_team']
        
        # Apply phase-specific success rates
        current_phase = self.determine_phase(year)
        success_rate = self.phases[current_phase].success_rate
        
        return available_surgeries * success_rate
    
    def calculate_unit_cost(self, cumulative_production: float) -> float:
        """Calculate unit cost with learning curve effects"""
        base_cost = self.costs.organ_processing
        learning_effect = self.calculate_learning_curve_effect(
            cumulative_production,
            base_rate=0.85  # 15% cost reduction per doubling
        )
        
        # Apply phase-specific cost multipliers
        current_phase = self.determine_phase(
            year=int(cumulative_production / self.capacity_params['initial'])
        )
        cost_multiplier = self.phases[current_phase].quality_control_factor
        
        return base_cost * learning_effect * cost_multiplier
    
    def determine_phase(self, year: int) -> str:
        """Determine current implementation phase"""
        if year < self.phases['experimental'].duration:
            return 'experimental'
        elif year < (self.phases['experimental'].duration + 
                    self.phases['scaling'].duration):
            return 'scaling'
        return 'mature'
    
    def calculate_learning_curve_effect(self, 
                                      volume: int,
                                      base_rate: float = 0.85) -> float:
        """Calculate learning curve cost reduction"""
        if volume <= 0:
            return 1.0
        return base_rate ** (np.log2(volume))
    
    def calculate_phase_specific_costs(self, 
                                     year: int,
                                     organ_type: str,
                                     production_volume: int) -> Dict[str, float]:
        """Calculate detailed costs for current phase"""
        current_phase = self.determine_phase(year)
        params = self.phases[current_phase]
        
        # Calculate costs
        costs = {
            'base_production': self.calculate_base_production_cost(production_volume),
            'regulatory': params.regulatory_costs,
            'training': params.training_requirements,
            'monitoring': params.monitoring_costs,
            'facility': self.calculate_facility_costs(current_phase, production_volume),
            'quality_control': self.calculate_qc_costs(current_phase, production_volume)
        }
        
        # Record all costs with phase information
        for category, amount in costs.items():
            self.cost_tracker.record_cost(
                phase=current_phase,
                category=category,
                amount=amount,
                year=year
            )
        
        return costs
    
    def record_initial_costs(self, initial_capacity: int):
        """Record initial setup costs"""
        self.cost_tracker.record_cost(
            phase='experimental',
            category='facility_construction',
            amount=self.costs.facility_construction,
            year=0
        )
        
        self.cost_tracker.record_cost(
            phase='experimental',
            category='initial_setup',
            amount=self.costs.facility_operation * 0.5,  # Half year of setup
            year=0
        )
        
        self.cost_tracker.record_cost(
            phase='experimental',
            category='regulatory_approval',
            amount=5_000_000,  # Initial regulatory costs
            year=0
        )
    
    def record_annual_costs(self, year: int, phase: str, 
                          production_volume: float, organ_type: str):
        """Record costs for each year of operation"""
        # Facility operations
        self.cost_tracker.record_cost(
            phase=phase,
            category='facility_operations',
            amount=self.costs.facility_operation,
            year=year
        )
        
        # Production costs
        unit_cost = self.calculate_unit_cost(production_volume)
        production_cost = unit_cost * production_volume
        self.cost_tracker.record_cost(
            phase=phase,
            category='production',
            amount=production_cost,
            year=year
        )
        
        # Phase-specific costs
        phase_params = self.phases[phase]
        self.cost_tracker.record_cost(
            phase=phase,
            category='regulatory',
            amount=phase_params.regulatory_costs,
            year=year
        )
        
        self.cost_tracker.record_cost(
            phase=phase,
            category='training',
            amount=phase_params.training_requirements,
            year=year
        )
        
        self.cost_tracker.record_cost(
            phase=phase,
            category='monitoring',
            amount=phase_params.monitoring_costs,
            year=year
        )
        
        # Quality control costs
        qc_cost = self.calculate_qc_costs(phase, production_volume)
        self.cost_tracker.record_cost(
            phase=phase,
            category='quality_control',
            amount=qc_cost,
            year=year
        )
    
    def calculate_surgical_teams(self, year: int) -> int:
        """Calculate number of surgical teams available in a given year"""
        params = self.surgical_params
        current_teams = params['initial_teams'] * (
            1 + params['growth_rate']
        ) ** year
        
        # Handle the case where max_teams is infinity
        if params['max_teams'] == float('inf'):
            return int(current_teams)
        else:
            return min(int(current_teams), int(params['max_teams']))
    
    def update_surgical_params(self, 
                             initial_teams: int = None,
                             growth_rate: float = None,
                             max_teams: int = None,
                             surgeries_per_team: int = None):
        """Update surgical parameters with new values"""
        if initial_teams is not None:
            self.surgical_params['initial_teams'] = initial_teams
        if growth_rate is not None:
            self.surgical_params['growth_rate'] = growth_rate
        if max_teams is not None:
            self.surgical_params['max_teams'] = float('inf') if max_teams == 0 else max_teams
        if surgeries_per_team is not None:
            self.surgical_params['surgeries_per_team'] = surgeries_per_team

class EnhancedXenoTransplantScaling:
    """Enhanced version of original XenoTransplantScaling"""
    def __init__(self, costs, regions, facility_model, growth_rates=None, 
                 initial_facilities=1, facility_growth_rate=0.25, 
                 max_facilities=None, max_surgical_teams=None):
        self.costs = costs
        self.regions = regions
        self.facility_model = facility_model
        self.growth_rates = growth_rates or {
            'deceased': 0.02,  # 2% annual growth
            'living': 0.015    # 1.5% annual growth
        }
        self.initial_facilities = initial_facilities
        self.facility_growth_rate = facility_growth_rate
        self.max_facilities = max_facilities
        self.max_surgical_teams = max_surgical_teams
        
        # Use OrganDemandData for organ demand information
        self.organ_demand = OrganDemandData.ORGAN_DEMAND

    def calculate_annual_demand(self, year: int, organ_type: str) -> float:
        """Calculate total annual demand for a specific organ type"""
        organ_data = self.organ_demand[organ_type]
        
        # Base demand is annual additions to waitlist
        base_demand = organ_data['annual_additions']
        
        # Apply growth rate
        growth_rate = organ_data['growth_rate']
        projected_demand = base_demand * (1 + growth_rate) ** year
        
        return projected_demand

    def calculate_current_supply(self, year: int, organ_type: str) -> float:
        """Calculate current (non-xeno) supply"""
        organ_data = self.organ_demand[organ_type]
        
        # Calculate growing traditional supply
        deceased = organ_data['deceased_annual'] * (1 + self.growth_rates['deceased']) ** year
        living = organ_data['living_annual'] * (1 + self.growth_rates['living']) ** year
        
        return deceased + living

    def calculate_supply_demand_gap(self, year: int, organ_type: str) -> Dict[str, float]:
        """Calculate gap between supply and demand"""
        demand = self.calculate_annual_demand(year, organ_type)
        current_supply = self.calculate_current_supply(year, organ_type)
        xeno_supply = self.project_organ_supply(year, organ_type)['organ_supply'].iloc[0]
        
        return {
            'demand': demand,
            'current_supply': current_supply,
            'xeno_supply': xeno_supply,
            'total_supply': current_supply + xeno_supply,
            'gap': demand - (current_supply + xeno_supply)
        }

    def project_organ_supply(self, year: int, organ_type: str) -> pd.DataFrame:
        """Project organ supply for a specific year"""
        print(f"\nCalculating supply for year {year}")  # Debug print
        
        # Calculate number of facilities for this year
        current_facilities = self.initial_facilities * (1 + self.facility_growth_rate) ** year
        print(f"Current facilities: {current_facilities}")  # Debug print
        
        # Calculate production capacity PER FACILITY
        per_facility_capacity = self.facility_model.calculate_annual_capacity(year)
        print(f"Per facility capacity: {per_facility_capacity}")  # Debug print
        
        # Total capacity across all facilities
        total_capacity = per_facility_capacity * current_facilities
        print(f"Total capacity: {total_capacity}")  # Debug print
        
        # Calculate surgical constraint
        available_surgeries = self.facility_model.calculate_available_surgeries(year, organ_type)
        print(f"Available surgeries: {available_surgeries}")  # Debug print
        
        # Take the minimum of capacity and surgical constraint
        organ_supply = min(total_capacity, available_surgeries)
        print(f"Final organ supply: {organ_supply}")  # Debug print
        
        # Create and return DataFrame
        result_df = pd.DataFrame({
            'year': [year],
            'facilities': [current_facilities],
            'per_facility_capacity': [per_facility_capacity],
            'total_capacity': [total_capacity],
            'available_surgeries': [available_surgeries],
            'organ_supply': [organ_supply]  # Make sure this column exists
        })
        
        print("DataFrame columns:", result_df.columns)  # Debug print
        return result_df

    def project_scaled_implementation(self, years: int, organ_type: str) -> pd.DataFrame:
        """Project implementation scaling over time"""
        results = []
        current_facilities = self.initial_facilities
        
        for year in range(years):
            # Calculate production capacity PER FACILITY
            per_facility_capacity = self.facility_model.calculate_annual_capacity(year)
            
            # Total capacity across all facilities
            total_capacity = per_facility_capacity * current_facilities
            
            # Calculate surgical constraint
            available_surgeries = self.facility_model.calculate_available_surgeries(year, organ_type)
            
            # Take the minimum of capacity and surgical constraint
            organ_supply = min(total_capacity, available_surgeries)
            
            # Calculate costs for this year
            unit_cost = self.facility_model.calculate_unit_cost(organ_supply)
            total_costs = unit_cost * organ_supply
            
            results.append({
                'year': year,
                'facilities': current_facilities,
                'per_facility_capacity': per_facility_capacity,
                'total_capacity': total_capacity,
                'available_surgeries': available_surgeries,
                'organ_supply': organ_supply,
                'unit_cost': unit_cost,
                'total_costs': total_costs,
                'cumulative_production': organ_supply if year == 0 else 
                    results[-1]['cumulative_production'] + organ_supply,
                'qalys_gained': organ_supply * 0.8  # Simplified QALY calculation
            })
            
            # Update facilities for next year using growth rate
            current_facilities *= (1 + self.facility_growth_rate)
        
        return pd.DataFrame(results)
    
    def project_all_transplant_types(self, years: int, organ_type: str) -> pd.DataFrame:
        """Project all types of transplants"""
        results = []
        organ_data = self.organ_demand[organ_type]
        
        for year in range(years):
            try:
                # Get supply data and extract organ supply
                supply_data = self.project_organ_supply(year, organ_type)
                xeno_supply = supply_data['organ_supply'].iloc[0]
                
                # Calculate traditional transplants with organ-specific baseline
                deceased = organ_data['deceased_annual'] * (1 + self.growth_rates['deceased']) ** year
                living = organ_data['living_annual'] * (1 + self.growth_rates['living']) ** year
                
                results.append({
                    'year': year,
                    'Xenotransplants': xeno_supply,
                    'Deceased Donor': deceased,
                    'Living Donor': living,
                    'Total': xeno_supply + deceased + living
                })
            except Exception as e:
                print(f"Error in year {year}: {str(e)}")
                raise
        
        return pd.DataFrame(results)
    
    def calculate_waitlist_deaths(self, years: int, organ_type: str) -> pd.DataFrame:
        """Calculate impact on waitlist deaths"""
        results = []
        organ_data = self.organ_demand[organ_type]
        baseline_waitlist = organ_data['waitlist_size']
        mortality_rate = organ_data['annual_deaths'] / organ_data['waitlist_size']
        
        for year in range(years):
            # Get total transplants for this year
            transplants = self.project_all_transplant_types(years, organ_type)
            total_transplants = transplants.loc[year, 'Total']
            
            # Calculate waitlist changes with organ-specific growth rate
            baseline = baseline_waitlist * (1 + organ_data['growth_rate']) ** year
            deaths_without_xeno = baseline * mortality_rate
            
            # Calculate reduced waitlist with xenotransplants
            with_xeno = max(0, baseline - total_transplants)
            deaths_with_xeno = with_xeno * mortality_rate
            
            results.append({
                'year': year,
                'baseline_waitlist': baseline,
                'waitlist_with_xeno': with_xeno,
                'lives_saved': deaths_without_xeno - deaths_with_xeno,
                'mortality_rate': mortality_rate
            })
        
        return pd.DataFrame(results)
    
    def calculate_traditional_supply(self, years: int, organ_type: str) -> List[float]:
        """Calculate traditional (non-xeno) supply projection"""
        organ_data = self.organ_demand[organ_type]
        return [
            (organ_data['deceased_annual'] * (1 + self.growth_rates['deceased'])**year +
             organ_data['living_annual'] * (1 + self.growth_rates['living'])**year)
            for year in range(years)
        ]
    
    def calculate_total_treatment_need(self, years: int, organ_type: str) -> List[float]:
        """Calculate total treatment need trajectory"""
        organ_data = self.organ_demand[organ_type]
        
        # Get baseline total treatment need and growth rate
        baseline_total = organ_data['total_treatment_baseline']
        growth_rate = organ_data['growth_rate']
        
        # Calculate yearly treatment needs
        total_treatment = []
        for year in range(years):
            # Calculate total treatment need excluding waitlist
            treatment_need = (baseline_total - organ_data['waitlist_size']) * (1 + growth_rate) ** year
            total_treatment.append(treatment_need)
        
        return total_treatment

    def project_implementation(self, years: int, initial_capacity: int,
                             mature_capacity: int, capacity_growth_rate: float,
                             initial_surgical_teams: int, surgical_team_growth_rate: float,
                             surgeries_per_team: int) -> pd.DataFrame:
        """Project implementation metrics over time"""
        facilities = []
        per_facility_capacity = []
        total_capacity = []
        available_surgeries = []
        organ_supply = []
        total_costs = []
        unit_costs = []
        
        for year in range(years):
            # Calculate number of facilities
            if self.max_facilities is not None:
                current_facilities = min(
                    self.max_facilities,
                    self.initial_facilities * (1 + self.facility_growth_rate) ** year
                ) if self.facility_growth_rate else self.max_facilities
            else:
                current_facilities = self.initial_facilities * (1 + self.facility_growth_rate) ** year
            
            facilities.append(current_facilities)
            
            # Calculate surgical teams
            if self.max_surgical_teams is not None:
                current_teams = min(
                    self.max_surgical_teams,
                    initial_surgical_teams * (1 + surgical_team_growth_rate) ** year
                ) if surgical_team_growth_rate else self.max_surgical_teams
            else:
                current_teams = initial_surgical_teams * (1 + surgical_team_growth_rate) ** year
            
            # Calculate per-facility capacity
            current_capacity = min(
                initial_capacity * (1 + capacity_growth_rate) ** year,
                mature_capacity
            )
            per_facility_capacity.append(current_capacity)
            
            # Calculate total production capacity
            current_total_capacity = current_facilities * current_capacity
            total_capacity.append(current_total_capacity)
            
            # Calculate available surgeries
            current_surgeries = current_teams * surgeries_per_team
            available_surgeries.append(current_surgeries)
            
            # Final organ supply is minimum of capacity and surgical capability
            current_supply = min(current_total_capacity, current_surgeries)
            organ_supply.append(current_supply)
            
            # Calculate costs
            unit_cost = self.facility_model.calculate_unit_cost(sum(organ_supply))
            total_cost = (
                current_facilities * self.costs.facility_operation +
                current_supply * (unit_cost + self.costs.transplant_procedure)
            )
            
            unit_costs.append(unit_cost)
            total_costs.append(total_cost)
        
        return pd.DataFrame({
            'year': range(years),
            'facilities': facilities,
            'surgical_teams': [initial_surgical_teams * (1 + surgical_team_growth_rate) ** year 
                             if not self.max_surgical_teams 
                             else min(self.max_surgical_teams, 
                                    initial_surgical_teams * (1 + surgical_team_growth_rate) ** year)
                             for year in range(years)],
            'per_facility_capacity': per_facility_capacity,
            'total_capacity': total_capacity,
            'available_surgeries': available_surgeries,
            'organ_supply': organ_supply,
            'unit_cost': unit_costs,
            'total_costs': total_costs
        })

    def calculate_summary_metrics(self, projections: pd.DataFrame) -> Dict[str, float]:
        """Calculate summary metrics from projections"""
        metrics = {
            'total_organs_produced': int(projections['organ_supply'].sum()),
            'peak_annual_production': int(projections['organ_supply'].max()),
            'final_annual_production': int(projections['organ_supply'].iloc[-1]),
            'total_facilities_needed': int(projections['facilities'].max()),
            'peak_unit_cost': float(projections['unit_cost'].max()),
            'final_unit_cost': float(projections['unit_cost'].iloc[-1]),
            'total_investment_required': float(projections['total_costs'].sum()),
            'peak_annual_cost': float(projections['total_costs'].max()),
            'final_annual_cost': float(projections['total_costs'].iloc[-1]),
            'average_unit_cost': float(
                projections['total_costs'].sum() / 
                projections['organ_supply'].sum()
            ),
            'surgical_capacity_utilization': float(
                (projections['organ_supply'] / 
                 projections['available_surgeries']).mean() * 100
            ),
            'production_capacity_utilization': float(
                (projections['organ_supply'] / 
                 projections['total_capacity']).mean() * 100
            )
        }
        
        # Add percentage changes
        metrics.update({
            'unit_cost_reduction': float(
                (metrics['final_unit_cost'] - metrics['peak_unit_cost']) / 
                metrics['peak_unit_cost'] * 100
            ),
            'production_growth': float(
                (metrics['final_annual_production'] / 
                 projections['organ_supply'].iloc[0] - 1) * 100
            )
        })
        
        return metrics

    def calculate_transplant_numbers(
        self,
        organ_type: str,
        years: int,
        projections: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate numbers for different types of transplants over time"""
        
        # Get base data for traditional transplants
        organ_data = self.organ_demand[organ_type]
        base_deceased = organ_data['deceased_annual']
        base_living = organ_data['living_annual']
        
        # Calculate traditional transplant growth
        transplant_data = []
        cumulative_xeno = 0  # Track cumulative xenotransplants
        
        for year in range(years):
            deceased = base_deceased * (1 + self.growth_rates['deceased']) ** year
            living = base_living * (1 + self.growth_rates['living']) ** year
            
            # Calculate system capacity as minimum of surgical and facility capacity
            surgical_capacity = projections['available_surgeries'].iloc[year]
            facility_capacity = projections['total_capacity'].iloc[year]
            system_capacity = min(surgical_capacity, facility_capacity)
            
            # Add this year's system capacity to cumulative total
            cumulative_xeno += system_capacity
            
            transplant_data.append({
                'year': year,
                'Deceased Donor': deceased,
                'Living Donor': living,
                'Xenotransplants': system_capacity,
                'Total': deceased + living + system_capacity,
                'Cumulative Xenotransplants': cumulative_xeno,
                'Total Supply': deceased + living + system_capacity,  # Include cumulative impact
                'Surgical Capacity': surgical_capacity,
                'Facility Capacity': facility_capacity
            })
        
        return pd.DataFrame(transplant_data)

    def calculate_waitlist_impact(self, organ_type: str, years: int, 
                                all_transplants: pd.DataFrame) -> pd.DataFrame:
        """Calculate impact on waitlist deaths"""
        organ_data = self.organ_demand[organ_type]
        waitlist_data = []
        
        # Get initial waitlist size and growth rate
        base_waitlist = organ_data['waitlist_size']
        growth_rate = organ_data['growth_rate']
        current_waitlist = base_waitlist  # Track current waitlist size
        
        for year in range(years):
            # Calculate baseline waitlist (without xeno)
            baseline = base_waitlist * (1 + growth_rate) ** year
            
            # Get this year's system capacity
            system_capacity = min(
                all_transplants['Surgical Capacity'].iloc[year],
                all_transplants['Facility Capacity'].iloc[year]
            )

            # Calculate new additions to waitlist this year
            new_additions = baseline - (base_waitlist * (1 + growth_rate) ** (year-1) if year > 0 else base_waitlist)

            # Update current waitlist: add new patients, subtract treated patients
            # current_waitlist = max(0, current_waitlist + new_additions - system_capacity)
            current_waitlist = max(0, baseline - system_capacity)

            waitlist_data.append({
                'year': year,
                'baseline_waitlist': baseline,
                'waitlist_with_xeno': current_waitlist,
                'annual_system_capacity': system_capacity,
                'new_additions': new_additions,
                'surgical_capacity': all_transplants['Surgical Capacity'].iloc[year],
                'facility_capacity': all_transplants['Facility Capacity'].iloc[year]
            })
        
        return pd.DataFrame(waitlist_data)

    def calculate_lives_saved(self, organ_type: str, years: int, 
                         waitlist_deaths: pd.DataFrame, 
                         all_transplants: pd.DataFrame) -> pd.DataFrame:
        """Calculate lives saved with priority given to patients at risk of death"""
        organ_data = self.organ_demand[organ_type]
        annual_waitlist_deaths = organ_data['annual_deaths']
        base_waitlist = organ_data['waitlist_size']
        
        # Map organ types to their end-stage disease abbreviations
        es_abbrev = {
            'kidney': 'esrd',   # End-Stage Renal Disease
            'heart': 'eshd',    # End-Stage Heart Disease
            'liver': 'esld',    # End-Stage Liver Disease
            'lung': 'espd',     # End-Stage Pulmonary Disease
            'pancreas': 'espad'  # End-Stage Pancreatic Disease (changed from espd)
        }
        
        # Get the correct end-stage disease abbreviation for this organ
        es_type = es_abbrev[organ_type]
        annual_es_deaths = organ_data[f'annual_{es_type}_deaths']
        es_mortality_rate = organ_data[f'{es_type}_mortality_rate']
        
        lives_saved_data = []
        cumulative_saved = 0
        cumulative_es_saved = 0
        
        for year in range(years):
            # Calculate baseline deaths (without xeno)
            baseline_deaths = annual_waitlist_deaths * (1 + organ_data['growth_rate']) ** year
            es_deaths = annual_es_deaths * (1 + organ_data['growth_rate']) ** year
            
            # Get system capacity for this year
            system_capacity = min(
                all_transplants['Surgical Capacity'].iloc[year],
                all_transplants['Facility Capacity'].iloc[year]
            )
            
            # First priority: save waitlist patients
            deaths_with_xeno = max(0, baseline_deaths - system_capacity)
            waitlist_lives_saved = baseline_deaths - deaths_with_xeno
            
            # Calculate remaining capacity after saving waitlist patients
            # remaining_capacity = max(0, system_capacity - waitlist_lives_saved)

            # Calculate remaining capacity after saving waitlist patients
            baseline = base_waitlist * (1 + organ_data['growth_rate']) ** year
            current_waitlist = max(0, baseline - system_capacity)
            remaining_capacity = max(0, system_capacity - baseline)

            print("system_capacity =" + str(system_capacity))
            print("current_waitlist =" + str(current_waitlist))
            print("base_waitlist =" + str(baseline))
            print("remaining_capacity =" + str(remaining_capacity))
            # Second priority: save end-stage disease patients (limited by both capacity and deaths)
            es_lives_saved = min(remaining_capacity, es_deaths)
            es_deaths_with_xeno = es_deaths - es_lives_saved
            print("es_deaths_with_xeno =" + str(es_deaths_with_xeno))
            print("es_lives_saved =" + str(es_lives_saved))

            # Update cumulative totals
            cumulative_saved += waitlist_lives_saved
            cumulative_es_saved += es_lives_saved
            
            lives_saved_data.append({
                'year': year,
                'baseline_deaths': baseline_deaths,
                'deaths_with_xeno': deaths_with_xeno,
                'waitlist_lives_saved': waitlist_lives_saved,
                'remaining_capacity': remaining_capacity,
                'es_baseline_deaths': es_deaths,
                'es_deaths_with_xeno': es_deaths_with_xeno,
                'es_lives_saved': es_lives_saved,
                'total_lives_saved': waitlist_lives_saved + es_lives_saved,
                'system_capacity': system_capacity,
                'cumulative_waitlist_saved': cumulative_saved,
                'cumulative_es_saved': cumulative_es_saved,
                'cumulative_total_saved': cumulative_saved + cumulative_es_saved
            })
        
        return pd.DataFrame(lives_saved_data)

def run_enhanced_analysis(
    organ_type: str,
    years: int,
    facility_growth_rate: float,
    max_facilities: int = None,
    initial_facilities: int = 1,
    initial_capacity: int = 100,
    mature_capacity: int = 500,
    capacity_growth_rate: float = 0.15,
    initial_surgical_teams: int = 5,
    surgical_team_growth_rate: float = 0.2,
    max_surgical_teams: int = None,
    surgeries_per_team: int = 24,
    growth_rates: Dict[str, float] = None
) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[float], List[float]]:
    """Run complete analysis with all parameters"""
    
    # Initialize models with non-None growth rates
    costs = Costs(
        facility_construction=100_000_000,
        facility_operation=20_000_000,
        organ_processing=50_000,
        transplant_procedure=250_000,
        training_per_team=500_000,
        cold_chain_per_mile=100
    )
    
    facility_model = EnhancedFacilityModel(costs=costs)
    regions = RegionalDistribution()
    
    # Initialize scaling model with modified parameters
    scaling_model = EnhancedXenoTransplantScaling(
        costs=costs,
        regions=regions,
        facility_model=facility_model,
        growth_rates=growth_rates,
        initial_facilities=initial_facilities,
        facility_growth_rate=facility_growth_rate,
        max_facilities=max_facilities,
        max_surgical_teams=max_surgical_teams
    )
    
    # Run projections with maximum values
    projections = scaling_model.project_implementation(
        years=years,
        initial_capacity=initial_capacity,
        mature_capacity=mature_capacity,
        capacity_growth_rate=capacity_growth_rate,
        initial_surgical_teams=initial_surgical_teams,
        surgical_team_growth_rate=surgical_team_growth_rate,
        surgeries_per_team=surgeries_per_team
    )
    
    # Calculate metrics including lives saved
    metrics = scaling_model.calculate_summary_metrics(projections)
    all_transplants = scaling_model.calculate_transplant_numbers(organ_type, years, projections)
    waitlist_deaths = scaling_model.calculate_waitlist_impact(organ_type, years, all_transplants)
    lives_saved = scaling_model.calculate_lives_saved(organ_type, years, waitlist_deaths, all_transplants)
    traditional_supply = scaling_model.calculate_traditional_supply(years, organ_type)
    total_treatment_need = scaling_model.calculate_total_treatment_need(years, organ_type)
    
    return (projections, metrics, all_transplants, waitlist_deaths, lives_saved,
            traditional_supply, total_treatment_need)

# Example usage:
if __name__ == "__main__":
    projections, metrics, all_transplants, waitlist_deaths, lives_saved, traditional_supply, total_treatment_need = run_enhanced_analysis('kidney', years=10, new_facilities_per_year=0.5)
    print("\nProjections:")
    print(projections)
    print("\nSummary Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: ${value:,.2f}")