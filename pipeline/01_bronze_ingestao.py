""""
Como rodar:
    docker compose run --rm app python pipeline/01_bronze_ingestao.py
"""

from __future__ import annotations

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from comum import criar_sessao, exigir_camada, glob_eventos, gravar_eventos, ler_eventos, titulo

# ---------------------------------------------------------------------------
# Schema explícito da landing (R4): nada de inferência.
#
# O contrato do evento (docs/CONTRATO_EVENTO.md) descreve um evento aninhado:
# campos de topo + objetos (page, traffic, device, params) + a lista items[].
# O `params` reúne chaves que variam por evento; declaramos aqui a união de todas
# elas, cada uma com o tipo do contrato. O que não existir num dado evento vem nulo.
# ---------------------------------------------------------------------------

SCHEMA_PAGE = StructType([
    StructField("location", StringType()),
    StructField("referrer", StringType()),
])

SCHEMA_TRAFFIC = StructType([
    StructField("source", StringType()),
    StructField("medium", StringType()),
    StructField("campaign", StringType()),
])

SCHEMA_DEVICE = StructType([
    StructField("category", StringType()),
    StructField("os", StringType()),
    StructField("browser", StringType()),
])

# União de todas as chaves de `params` descritas no contrato (seção "params por evento"),
# mais o element_id, que aparece nos eventos disparados por clique.
SCHEMA_PARAMS = StructType([
    # page_view
    StructField("page_title", StringType()),
    # scroll
    StructField("percent_scrolled", DoubleType()),
    # click / file_download
    StructField("link_url", StringType()),
    StructField("link_domain", StringType()),
    StructField("link_text", StringType()),
    StructField("link_id", StringType()),
    StructField("outbound", BooleanType()),
    # site_search
    StructField("search_term", StringType()),
    StructField("search_results", LongType()),
    StructField("search_area", StringType()),
    # video_*
    StructField("video_title", StringType()),
    StructField("video_provider", StringType()),
    StructField("video_duration", DoubleType()),
    StructField("video_current_time", DoubleType()),
    StructField("video_percent", DoubleType()),
    # file_download
    StructField("file_name", StringType()),
    StructField("file_extension", StringType()),
    # view_item_list / select_item
    StructField("item_list_id", StringType()),
    StructField("item_list_name", StringType()),
    # e-commerce com valor (view_item, add_to_cart, ... purchase, refund, generate_lead)
    StructField("currency", StringType()),
    StructField("value", DoubleType()),
    StructField("tax", DoubleType()),
    StructField("shipping", DoubleType()),
    # add_shipping_info
    StructField("shipping_tier", StringType()),
    # pagamento / transação
    StructField("payment_type", StringType()),
    StructField("transaction_id", StringType()),
    StructField("coupon", StringType()),
    # refund
    StructField("reason", StringType()),
    # generate_lead
    StructField("lead_source", StringType()),
    # sign_up / login
    StructField("method", StringType()),
    # eventos disparados por clique (opcional)
    StructField("element_id", StringType()),
])

SCHEMA_ITEM = StructType([
    StructField("item_id", StringType()),
    StructField("item_name", StringType()),
    StructField("item_category", StringType()),
    StructField("price", DoubleType()),
    StructField("quantity", LongType()),
])

# event_name, year, month e day vêm do CAMINHO das partições (ler_eventos as devolve
# como colunas), por isso NÃO entram no schema do JSON, que descreve só o conteúdo da linha.
SCHEMA_LANDING = StructType([
    StructField("event_id", StringType()),
    StructField("event_name", StringType()),
    StructField("event_ts", StringType()),
    StructField("user_pseudo_id", StringType()),
    StructField("user_id", StringType()),
    StructField("session_hint", StringType()),
    StructField("page", SCHEMA_PAGE),
    StructField("traffic", SCHEMA_TRAFFIC),
    StructField("device", SCHEMA_DEVICE),
    StructField("params", SCHEMA_PARAMS),
    StructField("items", ArrayType(SCHEMA_ITEM)),
])


def main() -> None:
    spark = criar_sessao("bronze")

    # R2: só roda se a landing existe; senão, para e diz o que rodar antes.
    exigir_camada(spark, glob_eventos("landing"), "python gerador/gerar_eventos.py")

    # R4: schema explícito, sem inferência. event_name/year/month/day vêm do caminho.
    landing = ler_eventos(spark, "landing", "json", schema=SCHEMA_LANDING)

    # R4: proveniência — de qual arquivo da landing a linha veio e quando foi lida.
    bronze = (
        landing
        .withColumn("arquivo_origem", F.input_file_name())
        .withColumn("ingerido_em", F.current_timestamp())
    )

    # R4: Bronze append-only, particionada por event_name/year/month/day, em Parquet.
    gravar_eventos(bronze, "bronze", "append")

    titulo("Bronze")
    total = bronze.count()
    print(f"linhas ingeridas: {total}")
    print("por event_name:")
    (bronze.groupBy("event_name")
        .count()
        .orderBy(F.col("count").desc())
        .show(50, truncate=False))

    spark.stop()


if __name__ == "__main__":
    main()
