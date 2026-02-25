class TMDBLLMService:

    def __init__(self, tmdb_service):
        self.tmdb = tmdb_service

    async def search_from_plan(self, plan: dict):
        params = {}

        if plan.get("query"):
            params["query"] = plan["query"]

        if plan.get("genres"):
            params["with_genres"] = self.tmdb.map_genres(plan["genres"])

        if plan.get("year_range"):
            params["primary_release_date.gte"] = f"{plan['year_range'][0]}-01-01"
            params["primary_release_date.lte"] = f"{plan['year_range'][1]}-12-31"

        if plan.get("min_rating"):
            params["vote_average.gte"] = plan["min_rating"]

        params["sort_by"] = plan.get("sort_by", "popularity.desc")

        return await self.tmdb.discover_movies(params, limit=plan.get("limit", 5))