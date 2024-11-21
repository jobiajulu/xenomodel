import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import (Costs, RegionalDistribution, EnhancedFacilityModel, 
                  EnhancedXenoTransplantScaling, run_enhanced_analysis)

st.set_page_config(
    page_title="Xenotransplantation Impact Model",
    page_icon="🫀",
    layout="wide"
)

def main():
    # Header
    st.title("🫀 Xenotransplantation Impact Model")
    
    # Sidebar for model parameters
    with st.sidebar:
        st.header("Model Parameters")
        
        # Core parameters
        organ_type = st.selectbox(
            "Organ Type",
            options=["kidney", "heart", "liver", "lung", "pancreas"],
            help="Select the type of organ for transplantation"
        )
        
        years = st.slider(
            "Projection Years",
            min_value=10,
            max_value=50,
            value=30,
            help="Number of years to project into the future"
        )
        
        # Facility parameters
        with st.expander("Facility Parameters", expanded=True):
            use_max_facilities = st.checkbox(
                "Set Maximum Number of Facilities",
                value=True,
                help="Cap the total number of facilities at a maximum value"
            )
            
            if use_max_facilities:
                max_facilities = st.number_input(
                    "Maximum Number of Facilities",
                    min_value=1,
                    value=500,
                    help="Maximum number of facilities, regardless of growth rate"
                )
            else:
                max_facilities = None
                
            facility_growth = st.slider(
                "Annual Facility Growth Rate (%)",
                min_value=0,
                max_value=100,
                value=25,
                help="How fast the number of facilities grows annually (until maximum if set)"
            ) / 100.0
            
            initial_facilities = st.number_input(
                "Initial Facilities",
                min_value=1,
                value=1,
                help="Number of production facilities at start"
            )
            
            initial_capacity = st.number_input(
                "Initial Annual Capacity per Facility",
                min_value=10,
                value=100,
                help="Number of organs each facility can produce initially"
            )
            
            mature_capacity = st.number_input(
                "Mature Annual Capacity per Facility",
                min_value=initial_capacity,
                value=500,
                help="Maximum number of organs each facility can produce when mature"
            )
            
            capacity_growth_rate = st.slider(
                "Per-Facility Capacity Growth Rate (%)",
                min_value=0,
                max_value=100,
                value=15,
                help="How fast each facility's production capacity grows annually"
            ) / 100.0
        
        # Surgical parameters
        with st.expander("Surgical Parameters", expanded=True):
            use_max_teams = st.checkbox(
                "Set Maximum Number of Surgical Teams",
                value=True,
                help="Cap the total number of surgical teams at a maximum value"
            )
            
            if use_max_teams:
                max_surgical_teams = st.number_input(
                    "Maximum Number of Surgical Teams",
                    min_value=1,
                    value=500,
                    help="Maximum number of surgical teams, regardless of growth rate"
                )
            else:
                max_surgical_teams = None
                
            surgical_team_growth = st.slider(
                "Surgical Team Growth Rate (%)",
                min_value=0,
                max_value=100,
                value=20,
                help="How fast the number of surgical teams grows annually (until maximum if set)"
            ) / 100.0
            
            initial_surgical_teams = st.number_input(
                "Initial Surgical Teams",
                min_value=1,
                value=5,
                help="Number of surgical teams at start"
            )
            
            surgeries_per_team = st.number_input(
                "Annual Surgeries per Team",
                min_value=1,
                value=450,
                help="Number of surgeries each team can perform annually"
            )
        
        # Cost parameters
        with st.expander("Cost Parameters", expanded=True):
            facility_cost = st.number_input(
                "Facility Construction (Millions USD)",
                min_value=10,
                value=100,
                help="Cost to build a new production facility"
            )
            
            facility_operation = st.number_input(
                "Annual Facility Operation (Millions USD)",
                min_value=1,
                value=20,
                help="Annual cost to operate each facility"
            )
            
            organ_processing = st.number_input(
                "Organ Processing Cost (USD)",
                min_value=1000,
                value=50000,
                step=1000,
                help="Cost to process each organ"
            )
            
            transplant_procedure = st.number_input(
                "Transplant Procedure Cost (USD)",
                min_value=10000,
                value=250000,
                step=10000,
                help="Cost of the transplant surgery"
            )
            
            training_cost = st.number_input(
                "Training Cost per Team (USD)",
                min_value=10000,
                value=500000,
                step=10000,
                help="Cost to train each surgical team"
            )
            
            cold_chain_cost = st.number_input(
                "Cold Chain Cost per Mile (USD)",
                min_value=1,
                value=100,
                help="Cost to transport organs per mile"
            )
        
        # Growth rates
        with st.expander("Growth Rate Parameters", expanded=True):
            growth_rates = {
                'deceased': st.slider(
                    "Deceased Donor Growth Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=2.0,
                    help="Annual growth rate for deceased donor transplants"
                ) / 100.0,
                'living': st.slider(
                    "Living Donor Growth Rate (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.5,
                    help="Annual growth rate for living donor transplants"
                ) / 100.0
            }
    
    # Initialize models and run analysis
    costs = Costs(
        facility_construction=facility_cost * 1_000_000,
        facility_operation=facility_operation * 1_000_000,
        organ_processing=organ_processing,
        transplant_procedure=transplant_procedure,
        training_per_team=training_cost,
        cold_chain_per_mile=cold_chain_cost
    )
    
    facility_model = EnhancedFacilityModel(costs=costs)
    
    # Run analysis with all user-defined parameters
    projections, metrics, all_transplants, waitlist_deaths, lives_saved, traditional_supply, total_treatment_need = run_enhanced_analysis(
        organ_type=organ_type,
        years=years,
        facility_growth_rate=facility_growth,
        max_facilities=max_facilities if use_max_facilities else None,
        initial_facilities=initial_facilities,
        initial_capacity=initial_capacity,
        mature_capacity=mature_capacity,
        capacity_growth_rate=capacity_growth_rate,
        initial_surgical_teams=initial_surgical_teams,
        surgical_team_growth_rate=surgical_team_growth,
        max_surgical_teams=max_surgical_teams if use_max_teams else None,
        surgeries_per_team=surgeries_per_team,
        growth_rates=growth_rates
    )

    # Create scaling model to calculate lives saved
    scaling_model = EnhancedXenoTransplantScaling(
        costs=costs,
        regions=RegionalDistribution(),
        facility_model=EnhancedFacilityModel(costs=costs)
    )
    
    # Calculate lives saved
    lives_saved = scaling_model.calculate_lives_saved(
        organ_type=organ_type,
        years=years,
        waitlist_deaths=waitlist_deaths,
        all_transplants=all_transplants
    )

    # Calculate demand trajectory
    organ_data = facility_model.organ_demand[organ_type]
    base_demand = organ_data['waitlist_size']
    growth_rate = organ_data['growth_rate']
    demand = [base_demand * (1 + growth_rate) ** year for year in range(years)]
    years_list = list(range(years))  # Create years list for plotting

    # Top-level metrics
    # First row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Lives Saved",
            f"{int(lives_saved['cumulative_total_saved'].iloc[-1]):,}",
            help="Cumulative number of lives saved through xenotransplantation"
        )
    
    with col2:
        st.metric(
            "Waitlist Lives Saved",
            f"{int(lives_saved['cumulative_waitlist_saved'].iloc[-1]):,}",
            help="Lives saved specifically from reducing waitlist deaths"
        )
    
    with col3:
        st.metric(
            "Additional Lives Saved",
            f"{int(lives_saved['cumulative_es_saved'].iloc[-1]):,}",
            help="Additional lives saved beyond waitlist reduction"
        )
    
    with col4:
        st.metric(
            "Current Annual Impact",
            f"{int(lives_saved['total_lives_saved'].iloc[-1]):,}/year",
            help="Current annual rate of lives being saved"
        )

    # Second row
    col1, col2, col3 = st.columns(3)
    with col1:
        if use_max_teams:
            final_teams = max_surgical_teams
        else:
            final_teams = int(initial_surgical_teams * (1 + surgical_team_growth) ** (years - 1))
        final_surgical_capacity = final_teams * surgeries_per_team
        st.metric(
            "Surgical Teams in Final Year",
            f"{final_teams:,} teams",
            delta=f"Can perform {final_surgical_capacity:,} surgeries/year",
            help="Number of trained surgical teams performing xenotransplants"
        )
    with col2:
        if use_max_facilities:
            final_facilities = max_facilities
        else:
            final_facilities = int(initial_facilities * (1 + facility_growth) ** (years - 1))
        final_facility_capacity = int(final_facilities * min(
            initial_capacity * (1 + capacity_growth_rate) ** (years - 1),
            mature_capacity
        ))
        st.metric(
            "Production Facilities in Final Year",
            f"{final_facilities:,} facilities",
            delta=f"Can produce {final_facility_capacity:,} organs/year",
            help="Number of active xenotransplant organ production facilities"
        )
    with col3:
        limiting_factor = "surgical teams" if final_surgical_capacity < final_facility_capacity else "facilities"
        bottleneck_amount = min(final_surgical_capacity, final_facility_capacity)
        excess_capacity = abs(final_surgical_capacity - final_facility_capacity)
        st.metric(
            f"System Capacity (Currently limited by {limiting_factor.title()})",
            f"{bottleneck_amount:,} xenotransplants/yr",
            delta=f"{excess_capacity:,} excess {limiting_factor} capacity needed",
            help=f"The {limiting_factor} are the bottleneck. Adding more {limiting_factor} would increase total system capacity."
        )

    # Create the tabs
    tab1, tab2, tab3 = st.tabs(["Impact Analysis", "Cost Impact", "Industrial Growth"])
    
    with tab1:  # Impact Analysis
        st.header("Impact Analysis")
        
        # 1. Transplant Type Projections
        st.subheader("1. Transplant Type Projections")
        st.write("""
        This graph shows the projected annual organ supply with and without xenotransplantation. 
        The traditional supply includes both deceased donor transplants (growing at 2% annually, modifiable in model) 
        and living donor transplants (growing at 1.5% annually, modifiable in model). The xenotransplant supply is based on 
        facility production capacity and available surgical teams. Base numbers are derived from OPTN/SRTR 
        2021 Annual Data Report.
        """)
        col1, col2 = st.columns(2)

        with col1:
            fig_transplant_types = go.Figure()
            
            # Add deceased donor transplants
            fig_transplant_types.add_trace(
                go.Scatter(
                    x=years_list,
                    y=all_transplants['Deceased Donor'],
                    name="Deceased Donor",
                    stackgroup='one',
                    line=dict(width=0.5)
                )
            )
            # Add living donor transplants
            fig_transplant_types.add_trace(
                go.Scatter(
                    x=years_list,
                    y=all_transplants['Living Donor'],
                    name="Living Donor",
                    stackgroup='one',
                    line=dict(width=0.5)
                )
            )
            # Add xenotransplants
            fig_transplant_types.add_trace(
                go.Scatter(
                    x=years_list,
                    y=all_transplants['Xenotransplants'],
                    name="Xenotransplants",
                    stackgroup='one',
                    line=dict(width=0.5)
                )
            )
            
            fig_transplant_types.update_layout(
                title="Annual Transplants by Type",
                xaxis_title="Years from Now",
                yaxis_title="Number of Transplants",
                hovermode='x unified'
            )
            st.plotly_chart(fig_transplant_types, use_container_width=True)

        with col2:
            transplant_types_df = pd.DataFrame({
                'Year': years_list,
                'Living Donor': all_transplants['Living Donor'],
                'Deceased Donor': all_transplants['Deceased Donor'],
                'Xenotransplants': all_transplants['Xenotransplants'],
                'Total': [t + x for t, x in zip(traditional_supply, all_transplants['Xenotransplants'])]
            })
            st.dataframe(
                transplant_types_df.style.format({
                    'Living Donor': '{:,.0f}',
                    'Deceased Donor': '{:,.0f}',
                    'Xenotransplants': '{:,.0f}',
                    'Total': '{:,.0f}'
                })
            )
        # 1. Organ Supply and Demand (Waitlist)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_waitlist = go.Figure()
            fig_waitlist.add_trace(
                go.Scatter(x=years_list, y=demand, name="Waitlist Demand")
            )
            fig_waitlist.add_trace(
                go.Scatter(x=years_list, y=traditional_supply, name="Traditional Supply")
            )
            fig_waitlist.add_trace(
                go.Scatter(x=years_list, y=all_transplants['Xenotransplants'], name="Xeno Supply")
            )
            fig_waitlist.add_trace(
                go.Scatter(x=years_list, y=all_transplants['Deceased Donor'] + all_transplants['Living Donor'] + all_transplants['Xenotransplants'], name="Total Supply")
            )
            fig_waitlist.update_layout(
                title="Waitlist Supply and Demand",
                xaxis_title="Years from Now",
                yaxis_title="Number of Patients/Organs",
                hovermode='x unified'
            )
            st.plotly_chart(fig_waitlist, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Year': years_list,
                    'Waitlist Demand': demand,
                    'Traditional Supply': traditional_supply,
                    'Xeno Supply': all_transplants['Xenotransplants'],
                    'Total Supply': [t + x for t, x in zip(traditional_supply, all_transplants['Xenotransplants'])]
                }).style.format({col: '{:,.0f}' for col in ['Waitlist Demand', 'Traditional Supply', 'Xeno Supply', 'Total Supply']})
            )

        # 2. Waitlist Mortality Impact
        st.subheader("2. Transplant Waitlist Impact")
        st.write("""
        This visualization shows the projected annual waitlist size and deaths on the organ waitlist. The baseline scenario 
        (without xenotransplantation) assumes current waitlist growth and waitlist mortality rates continue with demographic growth. 
        The impact scenario shows reduced deaths as xenotransplant availability increases. Mortality rates 
        and growth projections are based on OPTN/SRTR historical data (2012-2021).
        """)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_deaths = go.Figure()
            fig_deaths.add_trace(
                go.Scatter(
                    x=waitlist_deaths['year'],
                    y=waitlist_deaths['baseline_waitlist'],  # Changed from baseline_deaths
                    name="Without Xenotransplants"
                )
            )
            fig_deaths.add_trace(
                go.Scatter(
                    x=waitlist_deaths['year'],
                    y=waitlist_deaths['waitlist_with_xeno'],  # Changed from deaths_with_xeno
                    name="With Xenotransplants"
                )
            )
            fig_deaths.update_layout(
                title="Annual Waitlist Size",
                xaxis_title="Years from Now",
                yaxis_title="Number of Patients",
                hovermode='x unified'
            )
            st.plotly_chart(fig_deaths, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Year': waitlist_deaths['year'],
                    'Waitlist Without Xeno': waitlist_deaths['baseline_waitlist'],
                    'Waitlist With Xeno': waitlist_deaths['waitlist_with_xeno'],
                    'Reduction': waitlist_deaths['baseline_waitlist'] - waitlist_deaths['waitlist_with_xeno']
                }).style.format({col: '{:,.0f}' for col in [
                    'Waitlist Without Xeno', 'Waitlist With Xeno', 'Reduction'
                ]})
            )
        # Waitlist Mortality Impact (Corrected Version)
        col1, col2 = st.columns(2)

        with col1:
            fig_waitlist_impact = go.Figure()
            
            # Add baseline deaths (without xeno)
            fig_waitlist_impact.add_trace(
                go.Scatter(
                    x=lives_saved['year'],
                    y=lives_saved['baseline_deaths'],
                    name="Without Xenotransplantation",
                    line=dict(color='rgb(239, 85, 59)', dash='dot')
                )
            )
            
            # Add deaths with xeno
            fig_waitlist_impact.add_trace(
                go.Scatter(
                    x=lives_saved['year'],
                    y=lives_saved['deaths_with_xeno'],
                    name="With Xenotransplantation",
                    line=dict(color='rgb(99, 110, 250)')
                )
            )
            
            fig_waitlist_impact.update_layout(
                title="Annual Deaths on Waitlist",
                xaxis_title="Years from Now",
                yaxis_title="Number of Deaths",
                hovermode='x unified'
            )
            st.plotly_chart(fig_waitlist_impact, use_container_width=True)

        with col2:
            waitlist_impact_df = pd.DataFrame({
                'Year': lives_saved['year'],
                'Deaths Without Xeno': lives_saved['baseline_deaths'],
                'Deaths With Xeno': lives_saved['deaths_with_xeno'],
                'Lives Saved': lives_saved['waitlist_lives_saved']
            })
            st.dataframe(
                waitlist_impact_df.style.format({
                    'Deaths Without Xeno': '{:,.0f}',
                    'Deaths With Xeno': '{:,.0f}',
                    'Lives Saved': '{:,.0f}'
                })
            )
        # 4. End Stage Organ Deaths
        st.subheader("4. End Stage Disease Treatment and Mortality Impact")
        # 3. Closing the Organ Shortage
        st.write("""
        This graph demonstrates how xenotransplantation could help meet total organ demand over time. 
        The demand line includes both waitlist patients and estimated end-stage organ disease patients 
        (based on CDC prevalence data). The supply line combines traditional organ sources with projected 
        xenotransplant production capacity. The shaded area represents the gap being closed.
        """)
        col1, col2 = st.columns(2)

        with col1:
            fig_shortage = go.Figure()
            
            # Add annual demand
            fig_shortage.add_trace(
                go.Scatter(
                    x=years_list,
                    y=total_treatment_need,
                    name="Annual Demand for End Stage Organ Treatment",
                    line=dict(color='rgb(239, 85, 59)', dash='dot')
                )
            )
            
            # Add total supply (traditional + xeno)
            total_supply = [t + x for t, x in zip(traditional_supply, all_transplants['Xenotransplants'])]
            fig_shortage.add_trace(
                go.Scatter(
                    x=years_list,
                    y=total_supply,
                    name="Supply with Xenotransplantation",
                    fill='tonexty',
                    line=dict(color='rgb(99, 110, 250)')
                )
            )
            
            fig_shortage.update_layout(
                title="Annual Organ Supply vs Demand for End Stage Organ Treatment",
                xaxis_title="Years from Now",
                yaxis_title="Number of Organs",
                hovermode='x unified'
            )
            st.plotly_chart(fig_shortage, use_container_width=True)

        with col2:
            shortage_df = pd.DataFrame({
                'Year': years_list,
                'Annual Demand': total_treatment_need,
                'Total Supply': total_supply,
                'Shortage': [d - s for d, s in zip(total_treatment_need, total_supply)]
            })
            st.dataframe(
                shortage_df.style.format({
                    'Annual Demand for End Stage Organ Treatment': '{:,.0f}',
                    'Total Supply with Xenotransplantation': '{:,.0f}',
                    'Shortage': '{:,.0f}'
                })
            )
        st.markdown("""---""")
        st.write("""
        Beyond the impact on waitlist morality, this visualization shows the broader impact on end-stage organ disease mortality. 
        The model assumes that after meeting waitlist demand, remaining xenotransplant capacity can serve 
        additional patients with end-stage disease who never made it onto the waitlist. Disease prevalence 
        and mortality rates are sourced from CDC and NIH databases.
        """)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_es = go.Figure()
            fig_es.add_trace(
                go.Scatter(
                    x=lives_saved['year'],
                    y=lives_saved['es_baseline_deaths'],  # Changed from es_baseline_deaths
                    name="Without Xenotransplants"
                )
            )
            fig_es.add_trace(
                go.Scatter(
                    x=lives_saved['year'],
                    y=lives_saved['es_deaths_with_xeno'],  # Changed from es_deaths_with_xeno
                    name="With Xenotransplants"
                )
            )
            fig_es.update_layout(
                title="Annual End Stage Disease Deaths",
                xaxis_title="Years from Now",
                yaxis_title="Number of Deaths"
            )
            st.plotly_chart(fig_es, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Year': lives_saved['year'],
                    'Deaths Without Xeno': lives_saved['es_baseline_deaths'],
                    'Deaths With Xeno': lives_saved['es_deaths_with_xeno'],
                    'Lives Saved': lives_saved['es_lives_saved'] # Changed to match actual column name
                }).style.format({col: '{:,.0f}' for col in [
                    'Deaths Without Xeno', 'Deaths With Xeno', 'Lives Saved'
                ]})
            )
  
        # 5. Cumulative Lives Saved
        st.subheader("5. Cumulative Lives Saved")
        st.write("""
        This graph shows the total cumulative impact of xenotransplantation over time, combining both 
        waitlist patients saved and additional end-stage disease patients treated. The model accounts for 
        both immediate waitlist mortality prevention and broader access to transplantation for those who 
        would otherwise not receive treatment. Five-year survival rates post-transplant are based on SRTR data.
        """)
        col1, col2 = st.columns(2)
        
        with col1:
            fig_lives = go.Figure()
            fig_lives.add_trace(
                go.Scatter(x=lives_saved['year'], y=lives_saved['cumulative_waitlist_saved'], 
                          name="Waitlist Lives", stackgroup='one')
            )
            fig_lives.add_trace(
                go.Scatter(x=lives_saved['year'], y=lives_saved['cumulative_es_saved'], 
                          name="End Stage Lives", stackgroup='one')
            )
            fig_lives.update_layout(
                title="Cumulative Lives Saved",
                xaxis_title="Years from Now",
                yaxis_title="Number of Lives"
            )
            st.plotly_chart(fig_lives, use_container_width=True)
        
        with col2:
            st.dataframe(
                pd.DataFrame({
                    'Year': lives_saved['year'],
                    'Waitlist Lives': lives_saved['cumulative_waitlist_saved'],
                    'End Stage Lives': lives_saved['cumulative_es_saved'],
                    'Total Lives': lives_saved['cumulative_total_saved']
                }).style.format({col: '{:,.0f}' for col in ['Waitlist Lives', 'End Stage Lives', 'Total Lives']})
            )

        
    with tab2:  # Cost Impact
        st.header("Healthcare System Cost Impact")
        
        # Total Healthcare Cost Over Time
        fig_total_cost = go.Figure()
        fig_total_cost.add_trace(
            go.Scatter(
                x=projections['year'],
                y=projections['total_costs'],
                name="Total Healthcare Cost",
                fill='tozeroy'
            )
        )
        fig_total_cost.update_layout(
            title="Total Healthcare Cost Over Time",
            xaxis_title="Years from Now",
            yaxis_title="Cost (USD)",
            yaxis_tickformat='$,.0f'
        )
        st.plotly_chart(fig_total_cost, use_container_width=True)
        
        # Cost per Xenotransplant Over Time
        fig_unit_cost = go.Figure()
        fig_unit_cost.add_trace(
            go.Scatter(
                x=projections['year'],
                y=projections['unit_cost'],
                name="Cost per Xenotransplant",
                fill='tozeroy'
            )
        )
        fig_unit_cost.update_layout(
            title="Cost per Xenotransplant Over Time",
            xaxis_title="Years from Now",
            yaxis_title="Cost (USD)",
            yaxis_tickformat='$,.0f'
        )
        st.plotly_chart(fig_unit_cost, use_container_width=True)

    with tab3:  # Industrial Growth
        st.header("Infrastructure Development")
        
        # Surgical Teams Section
        st.subheader("Surgical Teams Growth")
        col1, col2 = st.columns(2)
        
        # Calculate surgical metrics
        surgical_data = pd.DataFrame({
            'Year': projections['year'],
            'Number of Surgical Teams': projections['surgical_teams'],
            'Total Surgical Capacity': projections['available_surgeries'],
            'Average Surgeries per Team': (
                projections['available_surgeries'] / projections['surgical_teams']
            ).round(1)
        })
        
        with col1:
            fig_surgical = go.Figure()
            fig_surgical.add_trace(
                go.Scatter(
                    x=surgical_data['Year'],
                    y=surgical_data['Number of Surgical Teams'],
                    name="Surgical Teams",
                    line=dict(color='rgb(100, 100, 255)', width=2)
                )
            )
            fig_surgical.update_layout(
                title="Growth in Surgical Teams",
                xaxis_title="Years from Now",
                yaxis_title="Number of Teams"
            )
            st.plotly_chart(fig_surgical, use_container_width=True)
        
        with col2:
            st.dataframe(
                surgical_data.style.format({
                    'Number of Surgical Teams': '{:,.0f}',
                    'Total Surgical Capacity': '{:,.0f}',
                    'Average Surgeries per Team': '{:.1f}'
                })
            )
        
        # Facilities Section
        st.subheader("Production Facilities Growth")
        col1, col2 = st.columns(2)
        
        # Calculate facility metrics
        facility_data = pd.DataFrame({
            'Year': projections['year'],
            'Number of Facilities': projections['facilities'],
            'Total Production Capacity': projections['total_capacity'],
            'Average Production per Facility': (
                projections['total_capacity'] / projections['facilities']
            ).round(1)
        })
        
        with col1:
            fig_facilities = go.Figure()
            fig_facilities.add_trace(
                go.Scatter(
                    x=facility_data['Year'],
                    y=facility_data['Number of Facilities'],
                    name="Facilities",
                    line=dict(color='rgb(100, 200, 100)', width=2)
                )
            )
            fig_facilities.update_layout(
                title="Growth in Production Facilities",
                xaxis_title="Years from Now",
                yaxis_title="Number of Facilities"
            )
            st.plotly_chart(fig_facilities, use_container_width=True)
        
        with col2:
            st.dataframe(
                facility_data.style.format({
                    'Number of Facilities': '{:,.0f}',
                    'Total Production Capacity': '{:,.0f}',
                    'Average Production per Facility': '{:.1f}'
                })
            )
        
        # Cost Analysis Section
        st.subheader("Development Costs")
        col1, col2 = st.columns(2)
        
        # Calculate development costs
        surgical_costs = projections['surgical_teams'] * costs.training_per_team
        facility_costs = (projections['facilities'] * 
                        (costs.facility_construction + costs.facility_operation))
        
        cost_data = pd.DataFrame({
            'Year': projections['year'],
            'Surgical Program Costs': surgical_costs,
            'Facility Development Costs': facility_costs,
            'Total Infrastructure Costs': surgical_costs + facility_costs
        })
        
        with col1:
            fig_costs = go.Figure()
            fig_costs.add_trace(
                go.Scatter(
                    x=cost_data['Year'],
                    y=cost_data['Surgical Program Costs'],
                    name="Surgical Program",
                    line=dict(color='rgb(100, 100, 255)', width=2)
                )
            )
            fig_costs.add_trace(
                go.Scatter(
                    x=cost_data['Year'],
                    y=cost_data['Facility Development Costs'],
                    name="Facility Development",
                    line=dict(color='rgb(100, 200, 100)', width=2)
                )
            )
            fig_costs.update_layout(
                title="Development Costs Over Time",
                xaxis_title="Years from Now",
                yaxis_title="Cost (USD)",
                yaxis_tickformat='$,.0f'
            )
            st.plotly_chart(fig_costs, use_container_width=True)
        
        with col2:
            st.dataframe(
                cost_data.style.format({
                    'Surgical Program Costs': '${:,.0f}',
                    'Facility Development Costs': '${:,.0f}',
                    'Total Infrastructure Costs': '${:,.0f}'
                })
            )

if __name__ == "__main__":
    main()