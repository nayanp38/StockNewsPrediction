from app.services.news_service import NewsService


def test_topics_from_query_maps_monetary_event() -> None:
    topics = NewsService._topics_from_query("FED cutting interest rates and signaling easier policy")
    selected = topics.split(",")
    assert "economy_monetary" in selected


def test_topics_from_query_maps_technology_event() -> None:
    topics = NewsService._topics_from_query("New AI chips and semiconductor software demand")
    selected = topics.split(",")
    assert "technology" in selected


def test_topics_from_query_uses_default_when_no_signal() -> None:
    topics = NewsService._topics_from_query("lorem ipsum placeholder text")
    assert topics == "economy_macro,financial_markets,finance"
