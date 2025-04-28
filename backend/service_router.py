import re
from predictor import predict_h2, predict_lcoh, plot_trend, compare_parameters, sensitivity_analysis
from optimizer import optimize
from emissions import calculate_emissions

class ServiceRouter:
    """
    Service router to determine if questions can be answered using our ML models
    or if they're outside the scope of this hydrogen production system.
    """
    
    def __init__(self):
        # Define patterns for different types of queries
        self.patterns = {
            'hydrogen_production': [
                r'(h2|hydrogen|h₂)\s+(production|output|yield)',
                r'(produce|generate|yield)\s+(h2|hydrogen|h₂)',
                r'how\s+much\s+(h2|hydrogen|h₂)',
                r'(what|predict|calculate|estimate)\s+(the)?\s+(h2|hydrogen|h₂)\s+(production|output|yield)',
                r'at\s+(\d+)\s*(°c|c|celsius|deg|degrees)',
                r'at\s+(\d+)\s*(bar|pressure)',
                r'biogas\s+(flow|input|mass)\s+(\d+)',
            ],
            'economic_analysis': [
                r'(lcoh|levelized cost|cost of hydrogen)',
                r'(capex|capital expenditure|capital cost|investment cost)',
                r'(opex|operating expenditure|operating cost)',
                r'(cost|price|economics|economic)',
                r'(electricity|power)\s+(cost|price)',
                r'(\$|usd|dollar)',
            ],
            'trend_visualization': [
                r'(trend|graph|plot|chart|visualize|visualisation|visualization)',
                r'(show|display)\s+(me)?\s+(the)?\s+(relationship|correlation)',
                r'(how)\s+(does|do)\s+(.*)\s+(affect|impact|influence)',
            ],
            'optimization': [
                r'(optimize|optimise|optimization|optimisation)',
                r'(best|optimal|optimum|maximize|minimise|maximize|minimize)',
                r'(most efficient|most effective)',
            ],
            'emissions': [
                r'(emission|emissions|co2|carbon dioxide|carbon)',
                r'(ghg|greenhouse gas)',
                r'(environmental impact|climate impact)',
                r'(carbon footprint|carbon intensity)',
            ],
            'scenario_comparison': [
                r'(compare|comparison|versus|vs|vs\.|difference between)',
                r'(scenario|scenarios|case|cases)',
                r'(what if|what-if)',
            ],
        }
    
    def is_relevant_query(self, query):
        """
        Determine if the query is relevant to our hydrogen production system
        """
        query = query.lower()
        for category, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return True, category
        
        # Explicit irrelevant topics
        irrelevant_patterns = [
            r'(weather|forecast)',
            r'(stock|stocks|market|markets|finance)',
            r'(sports|sport|game|games)',
            r'(movie|movies|film|films)',
            r'(news|politics|election)',
            r'(recipe|recipes|cook|cooking)',
            r'(travel|vacation|trip)',
            r'(music|song|songs)',
        ]
        
        for pattern in irrelevant_patterns:
            if re.search(pattern, query):
                return False, None
                
        # If we can't clearly determine, assume it might be relevant
        return True, "unknown"
    
    def get_irrelevant_response(self, query):
        """
        Generate a response for irrelevant queries
        """
        return {
            "error": "Sorry, I'm specifically designed to help with hydrogen production analysis, economic metrics, optimization, and emissions calculations for biogas reforming processes. I don't have information about other topics outside this domain.",
            "suggestion": "Try asking about hydrogen production prediction, LCOH analysis, optimization, or emissions calculations."
        }
        
    def route_query(self, query, params=None):
        """
        Route the query to the appropriate service based on relevance
        """
        is_relevant, category = self.is_relevant_query(query)
        
        if not is_relevant:
            return self.get_irrelevant_response(query)
            
        # For relevant queries, we'll handle them in the Copilot actions
        # but we could add direct ML model calling here for complex cases
        return {"status": "routed", "category": category}
