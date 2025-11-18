--
-- PostgreSQL database dump
--

\restrict cmDHo02N7NwNKbNhWcWZpUZEN3qc0g2oHsuHWMarJVvlvlk0agr5l6CSG81lKVp

-- Dumped from database version 18.1
-- Dumped by pg_dump version 18.0

-- Started on 2025-11-18 13:40:31

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 5033 (class 1262 OID 16389)
-- Name: Sent_db; Type: DATABASE; Schema: -; Owner: postgres
--

CREATE DATABASE "Sent_db" WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'English_United States.1252';


ALTER DATABASE "Sent_db" OWNER TO postgres;

\unrestrict cmDHo02N7NwNKbNhWcWZpUZEN3qc0g2oHsuHWMarJVvlvlk0agr5l6CSG81lKVp
\connect "Sent_db"
\restrict cmDHo02N7NwNKbNhWcWZpUZEN3qc0g2oHsuHWMarJVvlvlk0agr5l6CSG81lKVp

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 4 (class 2615 OID 2200)
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

CREATE SCHEMA public;


ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- TOC entry 5034 (class 0 OID 0)
-- Dependencies: 4
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 222 (class 1259 OID 16412)
-- Name: articles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.articles (
    article_id integer NOT NULL,
    link character varying NOT NULL,
    title character varying NOT NULL,
    publish_date timestamp without time zone NOT NULL,
    stock_id integer NOT NULL,
    compound_sentiment_score double precision
);


ALTER TABLE public.articles OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 16434)
-- Name: articles_article_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.articles ALTER COLUMN article_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.articles_article_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 221 (class 1259 OID 16399)
-- Name: daily_stock_price; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.daily_stock_price (
    dsp_id integer CONSTRAINT daily_stok_price_dsp_id_not_null NOT NULL,
    stock_id integer CONSTRAINT daily_stok_price_stock_id_not_null NOT NULL,
    open real CONSTRAINT daily_stok_price_open_not_null NOT NULL,
    close real CONSTRAINT daily_stok_price_close_not_null NOT NULL,
    high real CONSTRAINT daily_stok_price_high_not_null NOT NULL,
    low real CONSTRAINT daily_stok_price_low_not_null NOT NULL,
    volume integer CONSTRAINT daily_stok_price_volume_not_null NOT NULL,
    date timestamp without time zone CONSTRAINT daily_stok_price_date_not_null NOT NULL
);


ALTER TABLE public.daily_stock_price OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 16435)
-- Name: daily_stok_price_dsp_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.daily_stock_price ALTER COLUMN dsp_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.daily_stok_price_dsp_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 220 (class 1259 OID 16390)
-- Name: stock_list; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stock_list (
    stock_id integer NOT NULL,
    symbol character varying NOT NULL
);


ALTER TABLE public.stock_list OWNER TO postgres;

--
-- TOC entry 225 (class 1259 OID 16436)
-- Name: stock_list_stock_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

ALTER TABLE public.stock_list ALTER COLUMN stock_id ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME public.stock_list_stock_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);


--
-- TOC entry 4876 (class 2606 OID 16423)
-- Name: articles article_id; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT article_id PRIMARY KEY (article_id);


--
-- TOC entry 4872 (class 2606 OID 16444)
-- Name: daily_stock_price date_stock; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_stock_price
    ADD CONSTRAINT date_stock UNIQUE (date, stock_id);


--
-- TOC entry 4874 (class 2606 OID 16411)
-- Name: daily_stock_price dsp_id; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_stock_price
    ADD CONSTRAINT dsp_id PRIMARY KEY (dsp_id);


--
-- TOC entry 4878 (class 2606 OID 16442)
-- Name: articles link; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT link UNIQUE (link);


--
-- TOC entry 4868 (class 2606 OID 16398)
-- Name: stock_list stock_id; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_list
    ADD CONSTRAINT stock_id PRIMARY KEY (stock_id);


--
-- TOC entry 4870 (class 2606 OID 16438)
-- Name: stock_list symbol; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stock_list
    ADD CONSTRAINT symbol UNIQUE (symbol);


--
-- TOC entry 4880 (class 2606 OID 16424)
-- Name: articles stock_list_articles_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT stock_list_articles_fk FOREIGN KEY (stock_id) REFERENCES public.stock_list(stock_id);


--
-- TOC entry 4879 (class 2606 OID 16429)
-- Name: daily_stock_price stock_list_daily_stok_price_fk; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.daily_stock_price
    ADD CONSTRAINT stock_list_daily_stok_price_fk FOREIGN KEY (stock_id) REFERENCES public.stock_list(stock_id);


-- Completed on 2025-11-18 13:40:32

--
-- PostgreSQL database dump complete
--

\unrestrict cmDHo02N7NwNKbNhWcWZpUZEN3qc0g2oHsuHWMarJVvlvlk0agr5l6CSG81lKVp

