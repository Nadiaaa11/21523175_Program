import asyncio
import http
import logging
import os
import re
import shutil
import time
import traceback
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import aiomysql
import cloudinary
import cloudinary.uploader
import pandas as pd
import pymysql
import requests
import spacy
from databases import Database
from deep_translator import MyMemoryTranslator
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langdetect import detect, lang_detect_exception
from libretranslatepy import LibreTranslateAPI
from markdown import markdown
from openai import OpenAI
from pydantic import BaseModel
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from sklearn.preprocessing import StandardScaler
from slugify import slugify
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from sqlalchemy import Column, DateTime, Float, Integer, String, case, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.future import select

# Try importing SentenceTransformer and numpy
SENTENCE_TRANSFORMERS_AVAILABLE = False
SKLEARN_AVAILABLE = True 
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("WARNING: 'sentence-transformers' and/or 'numpy' not found. Semantic search will fallback to TF-IDF or basic word overlap.")
    import numpy as np 


# ================================
# FASHION CATEGORIES CONSTANTS
# ================================

class FashionCategories:
    """
    Centralized fashion categories with comprehensive synonym management
    """
    
    # CORE CLOTHING ITEMS
    CLOTHING_TERMS = [
        # TOPS
        'kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt',
        'sweater', 'cardigan', 'hoodie', 'tank top', 'crop top', 'tube top',
        'halter top', 'camisole', 'singlet', 'vest', 'rompi', 'polo shirt',
        'henley', 'turtleneck', 'off shoulder', 'cold shoulder', 'wrap top',
        
        # BOTTOMS
        'celana', 'pants', 'trousers', 'jeans', 'denim', 'rok', 'skirt',
        'shorts', 'leggings', 'jeggings',
        'palazzo pants', 'wide leg pants', 'skinny jeans', 'straight jeans', 'bootcut',
        'flare pants', 'culottes', 'palazzo', 'cargo pants', 'joggers',
        'track pants', 'sweatpants', 'chinos', 'capri', 'bermuda',

        # DRESSES
        'dress', 'gaun', 'terusan', 'maxi dress', 'mini dress', 'midi dress',
        'bodycon dress', 'a-line dress', 'shift dress', 'wrap dress',
        'slip dress', 'shirt dress', 'sweater dress', 'sundress',
        'cocktail dress', 'evening dress',
        
        # OUTERWEAR
        'jaket', 'jacket', 'blazer', 'coat', 'mantel',
        'bomber jacket', 'denim jacket', 'leather jacket', 'varsity jacket',
        'puffer jacket', 'windbreaker', 'raincoat', 'trench coat',
        'peacoat', 'parka', 'cape', 'poncho',

        # ACCESSORIES
        'shawl', 'pashmina', 'scarf', 'belt', 'bag', 'purse', 'jewelry',
        'necklace', 'earrings', 'bracelet', 'ring', 'watch',
    ]
    
    # SLEEVE TERMS
    SLEEVE_TERMS = [
        'lengan panjang', 'lengan pendek', 'long sleeve', 'long sleeves',
        'short sleeve', 'short sleeves', 'sleeveless', 'tanpa lengan',
        '3/4 sleeve', '3/4 sleeves', 'quarter sleeve', 'quarter sleeves',
        'cap sleeve', 'cap sleeves', 'bell sleeve', 'bell sleeves',
        'puff sleeve', 'puff sleeves', 'balloon sleeve', 'balloon sleeves',
        'bishop sleeve', 'bishop sleeves', 'dolman sleeve', 'dolman sleeves',
        'raglan sleeve', 'raglan sleeves', 'flutter sleeve', 'flutter sleeves'
    ]
    
    # FIT TERMS 
    FIT_TERMS = [
        'oversized', 'oversize', 'longgar', 'loose', 'baggy', 'relaxed',
        'fitted', 'ketat', 'tight', 'slim', 'skinny', 'regular fit',
        'tailored', 'structured', 'flowy', 'draped', 'a-line', 'straight'
    ]
    
    # LENGTH TERMS 
    LENGTH_TERMS = [
        'maxi', 'midi', 'mini', 'ankle length', 'knee length', 'thigh length',
        'floor length', 'tea length', 'above knee', 'below knee', 'cropped length',
        'cropped', 'crop', 'panjang', 'pendek', 'long', 'short'
    ]
    
    # NECKLINE TERMS 
    NECKLINE_TERMS = [
        'v-neck', 'scoop neck', 'crew neck', 'boat neck', 'off shoulder',
        'one shoulder', 'strapless', 'halter neck', 'high neck',
        'mock neck', 'cowl neck', 'square neck', 'sweetheart neck'
    ]
    
    # STYLE CATEGORIES 
    STYLE_TERMS = [
        'casual', 'santai', 'formal', 'resmi', 'elegant', 'elegan',
        'minimalis', 'minimalist', 'vintage', 'retro', 'bohemian', 'boho',
        'ethnic', 'etnik', 'modern', 'contemporary', 'classic', 'klasik',
        'trendy', 'fashionable', 'chic', 'sophisticated', 'edgy',
        'feminine', 'masculine', 'androgynous', 'romantic', 'sporty',
        'preppy', 'grunge', 'punk', 'gothic', 'kawaii', 'streetwear'
    ]
    
    # COLOR TERMS
    COLOR_TERMS = [
        'neutral', 'neutral colors', 'bright colors', 'pastel', 'pastels',
        'mixed', 'mixed colors', 'colorful', 'earth tones', 'natural colors',
        'warm colors', 'cool colors', 'monochrome', 'vibrant colors',
        'hitam', 'black', 'putih', 'white', 'merah', 'red', 'biru', 'blue',
        'hijau', 'green', 'kuning', 'yellow', 'orange', 'oranye',
        'ungu', 'purple', 'pink', 'merah muda', 'coklat', 'brown',
        'abu-abu', 'grey', 'gray', 'navy', 'biru tua', 'maroon',
        'burgundy', 'wine', 'cream', 'krem', 'beige', 'khaki',
        'gold', 'emas', 'silver', 'perak', 'rose gold', 'copper',
        'mint', 'turquoise', 'coral', 'salmon', 'lavender', 'lilac',
    ]
    
    # MATERIAL TERMS
    MATERIAL_TERMS = [
        'cotton', 'katun', 'silk', 'sutra', 'satin', 'chiffon',
        'lace', 'renda', 'denim', 'leather', 'kulit', 'faux leather',
        'velvet', 'beludru', 'corduroy', 'tweed', 'wool', 'wol',
        'cashmere', 'linen', 'polyester', 'spandex', 'elastane',
        'viscose', 'rayon', 'modal', 'bamboo', 'organic cotton'
    ]
    
    # PATTERN TERMS
    PATTERN_TERMS = [
        'polos', 'solid', 'plain', 'striped', 'garis-garis', 'polka dot',
        'floral', 'bunga-bunga', 'geometric', 'abstract', 'animal print',
        'leopard', 'zebra', 'snake print', 'plaid', 'checkered',
        'houndstooth', 'paisley', 'tribal', 'ethnic print', 'batik',
        'tie dye', 'ombre', 'gradient', 'metallic', 'glitter', 'sequin'
    ]
    
    # OCCASION TERMS
    OCCASION_TERMS = [
        'office', 'kantor', 'work', 'kerja', 'business', 'professional',
        'party', 'pesta', 'clubbing', 'nightout', 'date', 'kencan',
        'wedding', 'pernikahan', 'formal event', 'graduation', 'wisuda',
        'beach', 'pantai', 'vacation', 'liburan', 'travel', 'weekend',
        'everyday', 'sehari-hari', 'casual outing', 'shopping', 'hangout',
        'gym', 'workout', 'sport', 'olahraga', 'yoga', 'running',
    ]
    
    # GENDER TERMS for detection
    GENDER_TERMS = [
        'perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe',
        'pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo'
    ]
    
    # BLACKLISTED TERMS
    BLACKLISTED_TERMS = [
        'rb', 'ribu', 'jt', 'juta', '000', 'budget', 'anggaran', 'harga', 'price',
        'rupiah', 'rp', 'idr', 'cost', 'biaya', 'cm', 'kg', 'height', 'weight',
        'tinggi', 'berat', 'kulit', 'skin', 'yang', 'dan', 'atau', 'dengan',
        'untuk', 'dari', 'pada', 'akan', 'dapat', 'adalah', 'ini', 'itu',
        'saya', 'anda', 'kamu', 'mereka', 'dia', 'sangat', 'lebih', 'kurang',
        'baik', 'cocok', 'bisa', 'tolong', 'recommendation', 'rekomendasi',
        'suggestion', 'saran', 'preferensi', 'ukuran', 'ada', 'carikan',
        'tunjukkan', 'ingin', 'mau', 'cari', 'mencari', 'looking', 'for',
        'untuk', 'by', 'dengan', 'about', 'tentang', 'other', 'lain',
        'lainnya', 'additional', 'tambahan', 'semua', 'all', 'bagus',
        'great', 'nice', 'mantap'
    ]
    
    # CLOTHING CATEGORIES MAPPING
    CLOTHING_CATEGORIES = {
        'tops': ['kemeja', 'shirt', 'blouse', 'blus', 'atasan', 'kaos', 't-shirt', 'tshirt', 'sweater', 'hoodie', 'cardigan', 'blazer', 'tank', 'top'],
        'bottoms_pants': ['celana', 'pants', 'jeans', 'trousers', 'leggings'],
        'bottoms_skirts': ['rok', 'skirt'],
        'dresses': ['dress', 'gaun', 'terusan'],
        'outerwear': ['jaket', 'jacket', 'blazer', 'coat', 'mantel'],
        'shorts': ['shorts', 'celana pendek']
    }
    
    # ================================
    # CENTRALIZED SYNONYM MAPPINGS
    # ================================
    
    # COLOR SYNONYMS
    COLOR_SYNONYMS = {
        'black': ['hitam', 'dark', 'gelap', 'noir'],
        'hitam': ['black', 'dark', 'gelap', 'noir'],
        'white': ['putih', 'light', 'terang', 'blanc'],
        'putih': ['white', 'light', 'terang', 'blanc'],
        'red': ['merah', 'crimson', 'cherry', 'rouge'],
        'merah': ['red', 'crimson', 'cherry', 'rouge'],
        'blue': ['biru', 'navy', 'azure', 'bleu'],
        'biru': ['blue', 'navy', 'azure', 'bleu'],
        'green': ['hijau', 'lime', 'forest', 'vert'],
        'hijau': ['green', 'lime', 'forest', 'vert'],
        'yellow': ['kuning', 'gold', 'lemon', 'jaune'],
        'kuning': ['yellow', 'gold', 'lemon', 'jaune'],
        'grey': ['abu-abu', 'gray', 'silver', 'gris'],
        'abu-abu': ['grey', 'gray', 'silver', 'gris'],
        'gray': ['abu-abu', 'grey', 'silver', 'gris'],
        'brown': ['coklat', 'tan', 'bronze', 'brun'],
        'coklat': ['brown', 'tan', 'bronze', 'brun'],
        'pink': ['merah muda', 'rose', 'salmon', 'magenta'],
        'merah muda': ['pink', 'rose', 'salmon', 'magenta'],
        'navy': ['biru tua', 'dark blue', 'navy blue'],
        'biru tua': ['navy', 'dark blue', 'navy blue'],
        'orange': ['oranye', 'tangerine', 'amber'],
        'oranye': ['orange', 'tangerine', 'amber'],
        'purple': ['ungu', 'violet', 'lavender'],
        'ungu': ['purple', 'violet', 'lavender']
    }
    
    # SLEEVE SYNONYMS
    SLEEVE_SYNONYMS = {
        'long sleeve': ['lengan panjang', 'long sleeves', 'full sleeve', 'panjang'],
        'lengan panjang': ['long sleeve', 'long sleeves', 'full sleeve'],
        'short sleeve': ['lengan pendek', 'short sleeves', 'pendek'],
        'lengan pendek': ['short sleeve', 'short sleeves'],
        'sleeveless': ['tanpa lengan', 'tank', 'without sleeves', 'no sleeves'],
        'tanpa lengan': ['sleeveless', 'tank', 'without sleeves', 'no sleeves'],
        '3/4 sleeve': ['three quarter', '3/4 sleeves', 'quarter sleeve'],
        'quarter sleeve': ['3/4 sleeve', 'three quarter', '3/4 sleeves'],
        'cap sleeve': ['cap sleeves', 'short cap'],
        'bell sleeve': ['bell sleeves', 'flared sleeve'],
        'puff sleeve': ['puff sleeves', 'puffy sleeves']
    }
    
    # FIT SYNONYMS
    FIT_SYNONYMS = {
        'oversized': ['longgar', 'loose', 'baggy', 'oversize', 'boxy'],
        'longgar': ['oversized', 'loose', 'baggy', 'oversize', 'relaxed'],
        'loose': ['longgar', 'oversized', 'baggy', 'relaxed', 'flowing'],
        'tight': ['ketat', 'fitted', 'snug', 'close fitting'],
        'ketat': ['tight', 'fitted', 'snug', 'close fitting'],
        'slim': ['slim fit', 'fitted', 'slimfit', 'tailored'],
        'fitted': ['tight', 'ketat', 'slim', 'tailored', 'form fitting'],
        'regular': ['regular fit', 'standard', 'normal', 'classic fit'],
        'skinny': ['very slim', 'extra slim', 'super slim', 'ultra slim'],
        'relaxed': ['loose', 'comfortable', 'easy fit', 'casual fit'],
        'tailored': ['fitted', 'structured', 'sharp', 'precise']
    }
    
    # LENGTH SYNONYMS
    LENGTH_SYNONYMS = {
        'maxi': ['panjang', 'long', 'floor length', 'full length', 'ankle length'],
        'panjang': ['maxi', 'long', 'floor length', 'full length'],
        'long': ['panjang', 'maxi', 'floor length', 'full length'],
        'mini': ['pendek', 'short', 'above knee', 'very short'],
        'pendek': ['mini', 'short', 'above knee', 'very short'],
        'short': ['pendek', 'mini', 'above knee', 'brief'],
        'midi': ['medium', 'knee length', 'mid length', 'calf length'],
        'knee length': ['midi', 'medium', 'mid length'],
        'crop': ['cropped', 'short', 'pendek', 'cut off'],
        'cropped': ['crop', 'short', 'cut off', 'abbreviated'],
        'ankle length': ['maxi', 'long', 'full length'],
        'floor length': ['maxi', 'long', 'full length', 'gown length']
    }
    
    # STYLE SYNONYMS
    STYLE_SYNONYMS = {
        'casual': ['santai', 'kasual', 'relaxed', 'informal', 'laid back'],
        'santai': ['casual', 'kasual', 'relaxed', 'informal', 'comfortable'],
        'formal': ['resmi', 'dressy', 'business', 'professional', 'elegant'],
        'resmi': ['formal', 'dressy', 'business', 'professional', 'official'],
        'elegant': ['elegan', 'classy', 'sophisticated', 'refined', 'graceful'],
        'elegan': ['elegant', 'classy', 'sophisticated', 'refined'],
        'vintage': ['retro', 'classic', 'old school', 'throwback', 'antique'],
        'retro': ['vintage', 'classic', 'old school', 'throwback'],
        'modern': ['contemporary', 'current', 'trendy', 'up to date'],
        'contemporary': ['modern', 'current', 'present day', 'trendy'],
        'minimalist': ['minimalis', 'simple', 'clean', 'understated'],
        'minimalis': ['minimalist', 'simple', 'clean', 'basic'],
        'bohemian': ['boho', 'free spirited', 'artistic', 'unconventional'],
        'boho': ['bohemian', 'free spirited', 'artistic', 'hippie'],
        'chic': ['stylish', 'fashionable', 'trendy', 'smart'],
        'sophisticated': ['refined', 'polished', 'cultured', 'elegant']
    }
    
    # MATERIAL SYNONYMS
    MATERIAL_SYNONYMS = {
        'cotton': ['katun', '100% cotton', 'pure cotton', 'cotton blend'],
        'katun': ['cotton', '100% cotton', 'pure cotton'],
        'denim': ['jeans', 'jean', 'denim cotton', 'blue jeans'],
        'silk': ['sutra', 'pure silk', '100% silk', 'silk blend'],
        'sutra': ['silk', 'pure silk', '100% silk'],
        'leather': ['kulit', 'genuine leather', 'real leather', 'leather material'],
        'kulit': ['leather', 'genuine leather', 'real leather'],
        'wool': ['wol', 'woolen', 'wool blend', 'pure wool'],
        'wol': ['wool', 'woolen', 'wool blend'],
        'linen': ['flax', 'linen blend', 'pure linen'],
        'polyester': ['poly', 'synthetic', 'polyester blend'],
        'chiffon': ['light fabric', 'sheer fabric', 'flowing fabric'],
        'satin': ['silky', 'smooth fabric', 'lustrous fabric']
    }
    
    @classmethod
    def get_clothing_category(cls, keyword):
        """Get clothing category for a keyword"""
        keyword_lower = keyword.lower()
        for category, terms in cls.CLOTHING_CATEGORIES.items():
            if any(term in keyword_lower for term in terms):
                return category
        return None
    
    @classmethod
    def is_clothing_item(cls, keyword):
        """Check if keyword is a clothing item"""
        return any(term in keyword.lower() for term in cls.CLOTHING_TERMS)
    
    @classmethod
    def is_style_term(cls, keyword):
        """Check if keyword is a style term"""
        return any(term in keyword.lower() for term in cls.STYLE_TERMS)
    
    @classmethod
    def is_color_term(cls, keyword):
        """Check if keyword is a color term"""
        return any(term in keyword.lower() for term in cls.COLOR_TERMS)
    
    @classmethod
    def is_blacklisted(cls, keyword):
        """Check if keyword is blacklisted"""
        return any(term in keyword.lower() for term in cls.BLACKLISTED_TERMS)
    
    @classmethod
    def is_gender_term(cls, keyword):
        """Check if keyword is a gender term"""
        return any(term in keyword.lower() for term in cls.GENDER_TERMS)
    
    @classmethod
    def get_synonyms(cls, keyword, attribute_type):
        """
        Get synonyms for a keyword based on attribute type
        """
        keyword_lower = keyword.lower()
        
        synonym_maps = {
            'color': cls.COLOR_SYNONYMS,
            'sleeve': cls.SLEEVE_SYNONYMS,
            'fit': cls.FIT_SYNONYMS,
            'length': cls.LENGTH_SYNONYMS,
            'style': cls.STYLE_SYNONYMS,
            'material': cls.MATERIAL_SYNONYMS
        }
        
        if attribute_type in synonym_maps:
            synonym_map = synonym_maps[attribute_type]
            return synonym_map.get(keyword_lower, [])
        
        return []
    
    @classmethod
    def find_attribute_matches(cls, keyword, search_text, attribute_type):
        """
        Find matches for an attribute keyword in search text using synonyms
        """
        keyword_lower = keyword.lower()
        search_text_lower = search_text.lower()
        
        # Direct match
        if keyword_lower in search_text_lower:
            return True, "DIRECT", keyword_lower
        
        # Synonym match
        synonyms = cls.get_synonyms(keyword, attribute_type)
        for synonym in synonyms:
            if synonym.lower() in search_text_lower:
                return True, "SYNONYM", synonym
        
        # Partial match for compound terms (sleeve and length specific)
        if attribute_type in ['sleeve', 'length']:
            if cls._check_partial_match(keyword_lower, search_text_lower, attribute_type):
                return True, "PARTIAL", keyword_lower
        
        return False, "NO_MATCH", ""
    
    @classmethod
    def _check_partial_match(cls, keyword, search_text, attribute_type):
        """Check for partial matches in compound terms"""
        if attribute_type == 'sleeve':
            if 'sleeve' in keyword or 'lengan' in keyword:
                if 'long' in keyword or 'panjang' in keyword:
                    return 'long' in search_text or 'panjang' in search_text
                elif 'short' in keyword or 'pendek' in keyword:
                    return 'short' in search_text or 'pendek' in search_text
        
        elif attribute_type == 'length':
            # Check for length-related partial matches
            length_components = ['long', 'short', 'panjang', 'pendek', 'maxi', 'mini', 'midi']
            for component in length_components:
                if component in keyword and component in search_text:
                    return True
        
        return False

    @classmethod
    def get_category_priority(cls, keyword):
        """Get priority score based on category"""
        if cls.is_clothing_item(keyword):
            return 400
        elif any(term in keyword for term in (
            cls.SLEEVE_TERMS + cls.FIT_TERMS + cls.LENGTH_TERMS + cls.NECKLINE_TERMS
        )) or cls.is_style_term(keyword):
            return 325
        elif cls.is_color_term(keyword) or any(term in keyword.lower() for term in cls.MATERIAL_TERMS + cls.PATTERN_TERMS):
            return 250
        elif any(term in keyword.lower() for term in cls.OCCASION_TERMS):
            return 200
        else:
            return 100
        
# Initialize FashionCategories
fashion_categories = FashionCategories()

# ================================
# CONTEXT-AWARE KEYWORD EXTRACTION
# ================================

def extract_contextual_keywords(user_input: str, ai_response: str = None) -> List[Tuple[str, float]]:
    """
    Keyword extraction that preserves attribute-clothing relationships
    """
    
    if not user_input and not ai_response:
        return []
    
    is_multi_item = detect_multi_item_request(user_input)
    print(f"Multi-item request: {is_multi_item}")
    
    clothing_phrases = extract_clothing_phrases(user_input)
    print(f"Clothing phrases found: {clothing_phrases}")
    
    standalone_attributes = extract_standalone_attributes(user_input, clothing_phrases)
    print(f"Standalone attributes: {standalone_attributes}")
    
    ai_phrases = []
    if ai_response:
        ai_phrases = extract_clothing_phrases(ai_response)
        print(f"AI clothing phrases: {ai_phrases}")
    
    final_keywords = score_contextual_keywords(clothing_phrases, standalone_attributes, ai_phrases, is_multi_item)
    
    print(f"\nFINAL CONTEXTUAL KEYWORDS:")
    for i, (keyword, score) in enumerate(final_keywords[:15]):
        print(f"   {i+1:2d}. '{keyword}' → {score:.1f}")
    
    # Store the multi-item flag for later use
    extract_contextual_keywords.is_multi_item_request = is_multi_item
    
    return final_keywords[:15]

def extract_clothing_phrases(text: str) -> List[Dict]:
    """
    Extract clothing items with their directly attached modifiers
    """
    if not text:
        return []
    
    doc = nlp(text.lower())
    clothing_phrases = []
    
    # Define clothing terms and their categories
    clothing_terms = {
        'skirt': 'bottoms_skirts', 'rok': 'bottoms_skirts',
        'pants': 'bottoms_pants', 'celana': 'bottoms_pants', 'jeans': 'bottoms_pants',
        'shirt': 'tops', 'kemeja': 'tops', 'blouse': 'tops', 'blus': 'tops',
        'dress': 'dresses', 'gaun': 'dresses', 'terusan': 'dresses',
        'jacket': 'outerwear', 'jaket': 'outerwear', 'coat': 'outerwear',
        't-shirt': 'tops', 'kaos': 'tops', 'sweater': 'tops'
    }
    
    # Define modifier terms and their types
    modifier_terms = {
        # Length modifiers
        'short': 'length', 'pendek': 'length', 'mini': 'length',
        'long': 'length', 'panjang': 'length', 'maxi': 'length', 'midi': 'length',
        
        # Fit modifiers
        'tight': 'fit', 'ketat': 'fit', 'slim': 'fit',
        'loose': 'fit', 'longgar': 'fit', 'oversized': 'fit',
        
        # Sleeve modifiers
        'sleeveless': 'sleeve', 'tanpa lengan': 'sleeve',
        
        # Style modifiers
        'casual': 'style', 'formal': 'style', 'elegant': 'style',
        
        # Color modifiers
        'red': 'color', 'merah': 'color', 'blue': 'color', 'biru': 'color',
        'black': 'color', 'hitam': 'color', 'white': 'color', 'putih': 'color',
    }
    
    # Find clothing terms and look for nearby modifiers
    for token in doc:
        if token.text in clothing_terms:
            clothing_item = token.text
            category = clothing_terms[clothing_item]
            modifiers = []
            
            # Look for modifiers in a window around the clothing item
            start_idx = max(0, token.i - 3)  # Look 3 words before
            end_idx = min(len(doc), token.i + 2)  # Look 1 word after
            
            for i in range(start_idx, end_idx):
                if i != token.i:  # Skip the clothing item itself
                    modifier_token = doc[i]
                    if modifier_token.text in modifier_terms:
                        # Check if this modifier is close enough to be associated
                        distance = abs(i - token.i)
                        if distance <= 2:  # Within 2 words
                            modifiers.append(modifier_token.text)
            
            # Create the phrase
            if modifiers:
                phrase = ' '.join(modifiers + [clothing_item])
            else:
                phrase = clothing_item
            
            clothing_phrases.append({
                'phrase': phrase,
                'clothing': clothing_item,
                'modifiers': modifiers,
                'category': category,
                'position': token.i
            })
    
    # Handle compound phrases like "long sleeve shirt"
    clothing_phrases = merge_compound_phrases(clothing_phrases, text)
    
    return clothing_phrases

def merge_compound_phrases(phrases: List[Dict], original_text: str) -> List[Dict]:
    """
    Merge compound phrases like "long sleeve" with clothing items
    """
    compound_patterns = [
        (r'long\s+sleeve', 'long sleeve'),
        (r'short\s+sleeve', 'short sleeve'),
        (r'lengan\s+panjang', 'lengan panjang'),
        (r'lengan\s+pendek', 'lengan pendek'),
        (r'off\s+shoulder', 'off shoulder'),
        (r'button\s+up', 'button up'),
        (r'crew\s+neck', 'crew neck'),
        (r'v\s+neck', 'v neck')
    ]
    
    # Find compound modifiers in text
    compound_modifiers = []
    for pattern, compound in compound_patterns:
        matches = re.finditer(pattern, original_text.lower())
        for match in matches:
            compound_modifiers.append({
                'compound': compound,
                'start': match.start(),
                'end': match.end()
            })
    
    # Update phrases with compound modifiers
    for phrase_data in phrases:
        phrase_start = original_text.lower().find(phrase_data['phrase'])
        if phrase_start == -1:
            continue
            
        phrase_end = phrase_start + len(phrase_data['phrase'])
        
        # Check if any compound modifier is near this phrase
        for compound_data in compound_modifiers:
            # If compound is within reasonable distance of the clothing item
            distance = min(
                abs(compound_data['end'] - phrase_start),
                abs(compound_data['start'] - phrase_end)
            )
            
            if distance <= 20:  # Within 20 characters
                # Replace individual modifiers with compound
                compound_parts = compound_data['compound'].split()
                
                # Remove individual parts if they exist
                for part in compound_parts:
                    if part in phrase_data['modifiers']:
                        phrase_data['modifiers'].remove(part)
                
                # Add compound modifier
                if compound_data['compound'] not in phrase_data['modifiers']:
                    phrase_data['modifiers'].append(compound_data['compound'])
                
                # Update phrase
                if phrase_data['modifiers']:
                    phrase_data['phrase'] = ' '.join(phrase_data['modifiers'] + [phrase_data['clothing']])
    
    return phrases

def extract_standalone_attributes(text: str, clothing_phrases: List[Dict]) -> List[Dict]:
    """
    Extract attributes that aren't directly tied to specific clothing items
    These apply to the overall request or multiple items
    """
    if not text:
        return []
    
    # Words that are already part of clothing phrases
    used_words = set()
    for phrase_data in clothing_phrases:
        used_words.update(phrase_data['phrase'].split())
    
    standalone_attributes = []
    
    # Color terms that might apply globally
    color_terms = [
        'black', 'hitam', 'white', 'putih', 'red', 'merah', 'blue', 'biru',
        'green', 'hijau', 'yellow', 'kuning', 'pink', 'navy', 'grey', 'abu-abu'
    ]
    
    # Style terms that might apply globally
    style_terms = [
        'casual', 'santai', 'formal', 'resmi', 'elegant', 'elegan',
        'minimalist', 'minimalis', 'vintage', 'modern'
    ]
    
    # Occasion terms
    occasion_terms = [
        'office', 'kantor', 'work', 'kerja', 'business', 'professional',
        'party', 'pesta', 'clubbing', 'nightout', 'date', 'kencan',
        'wedding', 'pernikahan', 'formal event', 'graduation', 'wisuda',
        'beach', 'pantai', 'vacation', 'liburan', 'travel', 'weekend',
        'everyday', 'sehari-hari', 'casual outing', 'shopping', 'hangout',
        'gym', 'workout', 'sport', 'olahraga', 'yoga', 'running',
    ]
    
    all_standalone = color_terms + style_terms + occasion_terms
    
    doc = nlp(text.lower())
    for token in doc:
        if token.text in all_standalone and token.text not in used_words:
            attribute_type = 'color' if token.text in color_terms else \
                           'style' if token.text in style_terms else 'occasion'
            
            standalone_attributes.append({
                'attribute': token.text,
                'type': attribute_type,
                'position': token.i
            })
    
    return standalone_attributes

def score_contextual_keywords(clothing_phrases: List[Dict], standalone_attributes: List[Dict],
                            ai_phrases: List[Dict], is_multi_item: bool) -> List[Tuple[str, float]]:
    """
    Score the extracted contextual keywords
    """
    scored_keywords = []
    
    # Score clothing phrases (highest priority)
    for phrase_data in clothing_phrases:
        base_score = 400 
        
        if is_multi_item and phrase_data['modifiers']:
            base_score *= 1.5
        
        # Boost for specific modifiers
        modifier_boost = len(phrase_data['modifiers']) * 50
        
        final_score = base_score + modifier_boost
        scored_keywords.append((phrase_data['phrase'], final_score))
    
    for attr_data in standalone_attributes:
        base_score = 200 
        
        if attr_data['type'] == 'color':
            base_score = 250
        elif attr_data['type'] == 'style':
            base_score = 300
        
        scored_keywords.append((attr_data['attribute'], base_score))
    
    for phrase_data in ai_phrases:
        base_score = 150  
        scored_keywords.append((phrase_data['phrase'], base_score))

    scored_keywords.sort(key=lambda x: x[1], reverse=True)
    
    return scored_keywords

def detect_multi_item_request(user_input):
    """
    Detection for multi-item requests like "carikan kemeja dan celana" or "short pants and maxi skirt".
    """
    if not user_input:
        return False
        
    user_input_lower = user_input.lower().strip()
    
    print(f"\nMULTI-ITEM DETECTION DEBUG START: '{user_input}'")
    
    simple_responses = {
        "yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga", "1", "2", "3", "one", "two", "three", "satu", "dua", "tiga"
    }
    if user_input_lower in simple_responses:
        print(f"Simple response detected: '{user_input_lower}' - Returning False")
        return False
    
    multi_connectors = [r'\b(dan|and|atau|or|with|sama|plus|\+|&)\b']
    has_explicit_connector = any(re.search(pattern, user_input_lower) for pattern in multi_connectors)
    print(f"Has explicit connector: {has_explicit_connector}")

    clothing_categories_map = fashion_categories.CLOTHING_CATEGORIES
    
    found_clothing_terms_and_categories = []
    
    for category, terms in clothing_categories_map.items():
        for term in terms:
            if term in user_input_lower:
                found_clothing_terms_and_categories.append((category, term))
    
    distinct_clothing_categories_in_input = {cat for cat, _ in found_clothing_terms_and_categories}
    
    print(f"Distinct clothing categories in input: {distinct_clothing_categories_in_input} (Count: {len(distinct_clothing_categories_in_input)})")
    
    if has_explicit_connector and len(distinct_clothing_categories_in_input) >= 2:
        print(f"DECISION: Explicit connector AND 2+ distinct clothing categories - Returning True")
        return True
    
    if len(distinct_clothing_categories_in_input) >= 2:
        print(f"DECISION: 2+ distinct clothing categories - Returning True")
        return True
    
    if has_explicit_connector:
        distinct_clothing_terms_matched = {term for cat, term in found_clothing_terms_and_categories}
        if len(distinct_clothing_terms_matched) >= 2:
            print(f"DECISION: Explicit connector AND 2+ distinct clothing terms - Returning True")
            return True
    
    print(f"DECISION: No strong multi-item patterns found - Returning False")
    return False

# ================================
# LINKED KEYWORD SYSTEM
# ================================

class KeywordNode:
    """A node in the keyword linked list"""
    def __init__(self, keyword: str, weight: float, source: str, category: str):
        self.keyword = keyword
        self.weight = weight
        self.source = source  # 'user_input' or 'ai_response'
        self.category = category 
        self.timestamp = datetime.now().isoformat()
        self.mention_count = 1
        self.next = None 

class ClothingChain:
    """A linked list representing a clothing item and its attributes"""
    def __init__(self, clothing_item: str, weight: float, source: str):
        self.head = KeywordNode(clothing_item, weight, source, 'clothing_item')
        self.clothing_category = self._get_clothing_category(clothing_item)
        self.last_updated = datetime.now().isoformat()
        self.total_nodes = 1
    
    def _get_clothing_category(self, keyword: str):
        """Determine clothing category from keyword using FashionCategories"""
        return fashion_categories.get_clothing_category(keyword) or 'unknown'
    
    def add_attribute(self, keyword: str, weight: float, source: str, category: str):
        """Add a style attribute to this clothing chain with conflict resolution"""
        self._clean_conflicting_attributes(keyword)
        
        current = self.head
        while current:
            if current.keyword.lower() == keyword.lower():
                # Update existing keyword
                current.weight = max(current.weight, weight)
                current.mention_count += 1
                current.timestamp = datetime.now().isoformat()
                current.source = source
                self.last_updated = datetime.now().isoformat()
                return True
            current = current.next
        
        new_node = KeywordNode(keyword, weight, source, category)
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.total_nodes += 1
        self.last_updated = datetime.now().isoformat()
        return True
    
    def _clean_conflicting_attributes(self, new_keyword):
        """Remove conflicting attributes when adding new ones"""
        new_keyword_lower = new_keyword.lower()
        
        # Define conflict groups
        length_conflicts = {
            'long_terms': ['maxi', 'panjang', 'long', 'floor length'],
            'short_terms': ['mini', 'pendek', 'short', 'crop'],
            'medium_terms': ['midi', 'knee length', 'medium']
        }
        
        fit_conflicts = {
            'loose_terms': ['oversized', 'loose', 'longgar', 'baggy'],
            'tight_terms': ['slim', 'tight', 'ketat', 'fitted', 'skinny']
        }
        
        sleeve_conflicts = {
            'long_sleeve': ['lengan panjang', 'long sleeve', 'long sleeves'],
            'short_sleeve': ['lengan pendek', 'short sleeve', 'short sleeves'],
            'sleeveless': ['sleeveless', 'tanpa lengan']
        }
        
        all_conflicts = {**length_conflicts, **fit_conflicts, **sleeve_conflicts}
        
        # Find which group the new keyword belongs to
        new_keyword_group = None
        for group, terms in all_conflicts.items():
            if any(term in new_keyword_lower for term in terms):
                new_keyword_group = group
                break
        
        if not new_keyword_group:
            return  # Not a conflicting attribute
        
        # Find the conflict category (length, fit, or sleeve)
        conflict_category = None
        if new_keyword_group in length_conflicts:
            conflict_category = 'length'
            conflict_groups = length_conflicts
        elif new_keyword_group in fit_conflicts:
            conflict_category = 'fit'
            conflict_groups = fit_conflicts
        elif new_keyword_group in sleeve_conflicts:
            conflict_category = 'sleeve'
            conflict_groups = sleeve_conflicts
        
        # Remove conflicting attributes from the chain
        current_node = self.head.next
        prev_node = self.head
        
        while current_node:
            should_remove = False
            current_keyword_lower = current_node.keyword.lower()
            
            # Check if current node conflicts with new keyword
            for group, terms in conflict_groups.items():
                if group != new_keyword_group:  # Different group in same category
                    if any(term in current_keyword_lower for term in terms):
                        should_remove = True
                        print(f"Removing conflicting {conflict_category}: '{current_node.keyword}' (conflicts with '{new_keyword}')")
                        break
            
            if should_remove:
                prev_node.next = current_node.next
                self.total_nodes -= 1
                current_node = current_node.next
            else:
                prev_node = current_node
                current_node = current_node.next
    
    def get_all_keywords(self) -> List[Tuple[str, float]]:
        """Get all keywords in this chain as (keyword, weight) tuples"""
        keywords = []
        current = self.head
        while current:
            keywords.append((current.keyword, current.weight))
            current = current.next
        return keywords
    
    def apply_decay(self, decay_factor: float = 0.95):
        """Apply gentle decay to all weights in chain"""
        current = self.head
        while current:
            current.weight *= decay_factor
            current = current.next
        self.last_updated = datetime.now().isoformat()

class LinkedKeywordSystem:
    """Manages multiple clothing chains using linked lists"""
    
    def __init__(self):
        self.chains: Dict[str, ClothingChain] = {} 
        self.last_clothing_focus = None
        
    def _categorize_keyword(self, keyword: str) -> str:
        """Categorize a keyword using FashionCategories priority logic"""
        keyword_lower = keyword.lower()

        if fashion_categories.is_clothing_item(keyword):
            return 'clothing_item'
        elif any(term in keyword_lower for term in fashion_categories.SLEEVE_TERMS + fashion_categories.FIT_TERMS + fashion_categories.LENGTH_TERMS + fashion_categories.NECKLINE_TERMS):
            return 'attribute'
        elif (
            fashion_categories.is_style_term(keyword)
            or fashion_categories.is_color_term(keyword)
            or any(term in keyword_lower for term in fashion_categories.MATERIAL_TERMS + fashion_categories.PATTERN_TERMS)
        ):
            return 'style'
        elif any(term in keyword_lower for term in fashion_categories.OCCASION_TERMS):
            return 'occasion'
        else:
            return 'other'

    def _get_clothing_category_from_keyword(self, keyword: str) -> str:
        """Get the clothing category for a keyword using FashionCategories"""
        return fashion_categories.get_clothing_category(keyword) or 'unknown'
    
    def update_keywords(self, keywords: List[Tuple[str, float]], is_user_input: bool = False, is_multi_item_request: bool = False, clear_existing_chains: bool = False):
        source = "user_input" if is_user_input else "ai_response"

        if clear_existing_chains:
            self.chains.clear()

        if not keywords and not self.chains:
            return

        for keyword_phrase, weight in keywords:
            main_clothing_item = None
            main_clothing_category = None
            modifiers = []

            words = keyword_phrase.split()
            for word_idx, word in enumerate(words):
                if fashion_categories.is_clothing_item(word):
                    main_clothing_item = word
                    main_clothing_category = fashion_categories.get_clothing_category(word)
                    modifiers = [w for w_idx, w in enumerate(words) if w_idx != word_idx]
                    break

            if not main_clothing_item:
                if self.chains:
                    for category, chain in self.chains.items():
                        attr_category = self._categorize_keyword(keyword_phrase)
                        priority = fashion_categories.get_category_priority(keyword_phrase)
                        normalized = priority / 400.0
                        adjusted_weight = weight * normalized
                        chain.add_attribute(keyword_phrase, adjusted_weight, source, attr_category)
                continue

            if main_clothing_item and main_clothing_category and main_clothing_category != 'unknown':
                if main_clothing_category not in self.chains:
                    if is_user_input:
                        self.chains[main_clothing_category] = ClothingChain(main_clothing_item, weight, source)
                    else:
                        continue

                if main_clothing_category in self.chains:
                    chain = self.chains[main_clothing_category]
                    chain.head.weight = max(chain.head.weight, weight)
                    chain.head.mention_count += 1

                    for modifier in modifiers:
                        modifier_category = self._categorize_modifier(modifier)
                        priority = fashion_categories.get_category_priority(modifier)
                        normalized = priority / 400.0
                        adjusted_weight = weight * normalized
                        chain.add_attribute(modifier, adjusted_weight, source, modifier_category)

            self._apply_decay()
            self._print_current_state()

    def _categorize_modifier(self, modifier: str) -> str:
        """Categorize a modifier into its type"""
        modifier_lower = modifier.lower()
        
        if any(term in modifier_lower for term in fashion_categories.LENGTH_TERMS):
            return 'length'
        elif any(term in modifier_lower for term in fashion_categories.FIT_TERMS):
            return 'fit'
        elif any(term in modifier_lower for term in fashion_categories.SLEEVE_TERMS):
            return 'sleeve'
        elif fashion_categories.is_color_term(modifier):
            return 'color'
        elif fashion_categories.is_style_term(modifier):
            return 'style'
        else:
            return 'attribute'
    
    def _apply_decay(self, decay_factor: float = 0.95):
        for chain in self.chains.values():
            chain.apply_decay(decay_factor)
    
    def _print_current_state(self):
        """Print current state for debugging"""
        print(f"\nCURRENT KEYWORD CHAINS:")
        
        if not self.chains:
            print("   (No active chains)")
            return

        for clothing_category, chain in self.chains.items():
            keywords = chain.get_all_keywords()
            print(f"{clothing_category.upper()}: {len(keywords)} keywords")
            
            current_node = chain.head
            i = 0
            while current_node and i < 5:
                keyword = current_node.keyword
                weight = current_node.weight
                node_category = current_node.category
                node_source = current_node.source

                if i == 0:
                    print(f"HEAD: '{keyword}' → {weight:.1f} (Category: {node_category}, Source: {node_source})")
                else:
                    print(f"'{keyword}' → {weight:.1f} (Category: {node_category}, Source: {node_source})")
                
                current_node = current_node.next
                i += 1
            
            if len(keywords) > 5:
                print(f"and {len(keywords) - 5} more")

    def get_flattened_keywords(self) -> List[Tuple[str, float]]:
        """Get all keywords from all chains as flat list for product search"""
        all_keywords = []
        
        for chain in self.chains.values():
            all_keywords.extend(chain.get_all_keywords())
        
        all_keywords.sort(key=lambda x: x[1], reverse=True)
        
        return all_keywords

def get_keywords_for_product_search(user_context: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Get flattened keywords for product search"""
    if 'linked_keyword_system' not in user_context:
        return []
    
    return user_context['linked_keyword_system'].get_flattened_keywords()

def update_linked_keywords_contextual(user_context: Dict[str, Any], contextual_keywords: List[Tuple[str, float]],
                                    is_user_input: bool = False, is_multi_item_request: bool = False,
                                    clear_existing_chains: bool = False): 
    if 'linked_keyword_system' not in user_context:
        user_context['linked_keyword_system'] = LinkedKeywordSystem()
    
    user_context['linked_keyword_system'].update_keywords(contextual_keywords, is_user_input, is_multi_item_request, clear_existing_chains=clear_existing_chains)
                                                                                            
# ================================
# PRODUCT SEARCH AND TFIDF
# ================================

tfidf_vectorizer = None
TFIDF_MODEL_FITTED = False
product_tfidf_matrix = None

def preprocess_text_for_tfidf(text):
    """Preprocess text for better TF-IDF results"""
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def create_enhanced_product_text_corpus(all_products):
    """Create enhanced text corpus with better attribute emphasis using centralized categories"""
    corpus = []
    
    for product_row in all_products:
        product_name = preprocess_text_for_tfidf(product_row[1])
        product_detail = preprocess_text_for_tfidf(product_row[2])
        available_colors = preprocess_text_for_tfidf(product_row[7] if len(product_row) > 7 and product_row[7] else "")
        
        color_emphasis = f"{available_colors} {available_colors} {available_colors}" if available_colors else ""
        
        name_emphasis = f"{product_name} {product_name}"
        
        combined_text = f"{name_emphasis} {product_detail} {color_emphasis}"
        corpus.append(combined_text)
    
    return corpus

def initialize_tfidf_model(all_products):
    """Initialize TF-IDF model with attribute processing using centralized categories"""
    global tfidf_vectorizer, TFIDF_MODEL_FITTED, product_tfidf_matrix
    
    try:
        print("Initializing enhanced TF-IDF model with centralized attribute processing")
        
        product_texts = create_enhanced_product_text_corpus(all_products)
        
        if not product_texts:
            print("No product texts found for TF-IDF")
            return False
        
        tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.90,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            sublinear_tf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        product_tfidf_matrix = tfidf_vectorizer.fit_transform(product_texts)
        TFIDF_MODEL_FITTED = True
        
        print(f"Enhanced TF-IDF model with centralized attributes fitted successfully!")
        print(f"Vocabulary size: {len(tfidf_vectorizer.get_feature_names_out())}")
        print(f"Products indexed: {len(product_texts)}")
        print(f"Centralized synonyms: ENABLED")
        print(f"Enhanced attribute processing: ACTIVE")
        
        return True
        
    except Exception as e:
        print(f"Error initializing enhanced TF-IDF model: {str(e)}")
        TFIDF_MODEL_FITTED = False
        return False

def _calculate_raw_score_components(product_row, product_index, keywords, **kwargs):
    """
    Calculates the raw, unscaled score components (embedding, TF-IDF, keyword) for a single product.
    """
    global tfidf_vectorizer, TFIDF_MODEL_FITTED, enhanced_matcher, semantic_system

    raw_tfidf_similarity = 0.0
    if TFIDF_MODEL_FITTED and product_index is not None and product_index < product_tfidf_matrix.shape[0]:
        query_string = " ".join([kw for kw, _ in keywords])
        query_tfidf = tfidf_vectorizer.transform([preprocess_text_for_tfidf(query_string)])
        similarity = sk_cosine_similarity(query_tfidf, product_tfidf_matrix[product_index:product_index+1])[0][0]
        raw_tfidf_similarity = float(similarity)

    raw_embedding_similarity = 0.0
    if semantic_system.model is not None and product_row[0] in enhanced_matcher.product_embeddings:
        query_for_embedding = " ".join([kw for kw, _ in keywords])
        query_emb = semantic_system.get_semantic_embedding(query_for_embedding)
        product_emb = enhanced_matcher.product_embeddings[product_row[0]]
        if np.linalg.norm(query_emb) > 0 and np.linalg.norm(product_emb) > 0:
            similarity = np.dot(query_emb, product_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(product_emb))
            raw_embedding_similarity = float(similarity)

    product_name = product_row[1].lower()
    product_detail = product_row[2].lower()
    available_colors = product_row[7].lower() if len(product_row) > 7 and product_row[7] else ""
    search_text = f"{product_name} {product_detail} {available_colors}"

    clothing_bonus = 6.0
    attribute_bonus = 4.0

    raw_keyword_score = 0.0
    for keyword, weight in keywords:
        kw_lower = keyword.lower()
        if kw_lower in search_text:
            base_score = weight
            if fashion_categories.is_clothing_item(kw_lower):
                base_score *= clothing_bonus
            else:
                base_score *= attribute_bonus
            raw_keyword_score += base_score

    return raw_embedding_similarity, raw_tfidf_similarity, raw_keyword_score

def calculate_relevance_score(scaled_scores, product_info, focus_category=None):
    """
    Calculates the final weighted relevance score for a single product using 
    standardized scores and applying any relevant boosts.
    """

    embedding_weight = 0.5
    tfidf_weight = 0.2
    keyword_weight = 0.4
    main_item_boost = 1.5  

    scaled_emb, scaled_tfidf, scaled_kw = scaled_scores
    
    final_relevance_score = (
        (scaled_emb * embedding_weight) +
        (scaled_tfidf * tfidf_weight) +
        (scaled_kw * keyword_weight)
    )
    
    product_name = product_info[1]
    if focus_category and focus_category in str(product_name).lower():
        final_relevance_score += main_item_boost
        
    return final_relevance_score

async def fetch_products_from_db(db: AsyncSession, top_keywords: list, max_results=15, gender_category=None, budget_range=None, focus_category=None, is_multi_item_request=False, **kwargs):
    """
    Fetches and ranks products 
    """
    async def _query_and_rank(current_budget_range):

        variant_subquery = (
            select(
                ProductVariant.product_id,
                func.min(ProductVariant.product_price).label('min_price'),
                func.group_concat(ProductVariant.size.distinct()).label('available_sizes'),
                func.group_concat(ProductVariant.color.distinct()).label('available_colors'),
                func.sum(ProductVariant.stock).label('total_stock')
            ).where(ProductVariant.stock > 0).group_by(ProductVariant.product_id).subquery()
        )
        base_query = (
            select(
                Product.product_id, Product.product_name, Product.product_detail, Product.product_seourl,
                Product.product_gender, variant_subquery.c.min_price, variant_subquery.c.available_sizes,
                variant_subquery.c.available_colors, variant_subquery.c.total_stock, ProductPhoto.productphoto_path
            ).select_from(Product).join(variant_subquery, Product.product_id == variant_subquery.c.product_id).join(ProductPhoto, Product.product_id == ProductPhoto.product_id).where(variant_subquery.c.total_stock > 0)
        )
        if gender_category: base_query = base_query.where(Product.product_gender == gender_category.lower())
        if current_budget_range and len(current_budget_range) == 2:
            min_price, max_price = current_budget_range
            if min_price and max_price: base_query = base_query.where(variant_subquery.c.min_price.between(min_price, max_price))
            elif max_price: base_query = base_query.where(variant_subquery.c.min_price <= max_price)
            elif min_price: base_query = base_query.where(variant_subquery.c.min_price >= min_price)

        result = await db.execute(base_query)
        all_products_db = result.fetchall()

        if not all_products_db:
            return pd.DataFrame()

        def get_attribute_type(keyword):
            kw_lower = keyword.lower()
            if fashion_categories.is_clothing_item(kw_lower): return 'clothing'
            if fashion_categories.is_color_term(kw_lower): return 'color'
            if any(term in kw_lower for term in fashion_categories.FIT_TERMS): return 'fit'
            if any(term in kw_lower for term in fashion_categories.LENGTH_TERMS): return 'length'
            if any(term in kw_lower for term in fashion_categories.SLEEVE_TERMS): return 'sleeve'
            if fashion_categories.is_style_term(kw_lower): return 'style'
            if any(term in kw_lower for term in fashion_categories.MATERIAL_TERMS): return 'material'
            return 'other'

        mandatory_requirements = [
            {'keyword': kw.lower(), 'type': get_attribute_type(kw)}
            for kw, weight in top_keywords 
            if fashion_categories.get_category_priority(kw) >= 250
        ]

        unique_reqs = []
        seen_keywords = set()
        for req in mandatory_requirements:
            if req['keyword'] not in seen_keywords:
                unique_reqs.append(req)
                seen_keywords.add(req['keyword'])
        mandatory_requirements = unique_reqs

        if mandatory_requirements:
            logging.info(f"Applying Smart Checklist with requirements: {mandatory_requirements}")

            qualified_products = []
            for product_row in all_products_db:
                product_text = f"{product_row[1].lower()} {product_row[2].lower()} {product_row[7].lower()}"
                
                is_fully_qualified = True
                for req in mandatory_requirements:
                    keyword = req['keyword']
                    attr_type = req['type']

                    found, match_type, matched_term = fashion_categories.find_attribute_matches(keyword, product_text, attr_type)
                    
                    if not found:
                        is_fully_qualified = False
                        break

                if is_fully_qualified:
                    qualified_products.append(product_row)
            
            logging.info(f"Found {len(qualified_products)} of {len(all_products_db)} products that passed the Smart Checklist.")
            
            if not qualified_products:
                return pd.DataFrame()
            
            products_to_rank = qualified_products
        else:
            products_to_rank = all_products_db

        product_score_data = []
        for i, product_row in enumerate(products_to_rank):
            emb_score, tfidf_score, kw_score = _calculate_raw_score_components(product_row, i, top_keywords, **kwargs)
            product_score_data.append({"product_info": product_row, "scores": [emb_score, tfidf_score, kw_score]})

        if not product_score_data:
            return pd.DataFrame()

        raw_scores_array = np.array([item['scores'] for item in product_score_data])
        scaled_scores = StandardScaler().fit_transform(raw_scores_array) if raw_scores_array.shape[0] > 1 else raw_scores_array

        all_products_with_scores = []
        for i, item in enumerate(product_score_data):
            product_row = item['product_info']

            final_relevance_score = calculate_relevance_score(scaled_scores[i], product_row, focus_category)
            
            all_products_with_scores.append({
                "product_id": product_row[0], "product": product_row[1], "description": product_row[2],
                "price": product_row[5], "size": product_row[6], "color": product_row[7], "stock": product_row[8],
                "link": f"http://localhost/e-commerce-main/product-{product_row[3]}-{product_row[0]}",
                "photo": product_row[9], "relevance": final_relevance_score,
            })

        all_products_with_scores.sort(key=lambda x: x['relevance'], reverse=True)
        return pd.DataFrame(all_products_with_scores[:max_results])

    try:
        logging.info(f"Fetching products with keywords: {top_keywords}")
        if budget_range == (None, None) or not any(budget_range or []):
            budget_range = None

        if budget_range:
            logging.info(f"Searching with budget constraint: {budget_range}")
            products_df = await _query_and_rank(current_budget_range=budget_range)
            if not products_df.empty: return products_df, "within_budget"
            logging.info("No products in budget. Falling back to a general search.")
            products_df = await _query_and_rank(current_budget_range=None)
            status = "no_products_in_budget" if not products_df.empty else "no_products_found"
            return products_df, status
        else:
            logging.info("No budget specified, searching normally.")
            products_df = await _query_and_rank(current_budget_range=None)
            status = "no_budget_specified" if not products_df.empty else "no_products_found"
            return products_df, status

    except Exception as e:
        logging.error(f"Error in unified product fetching: {e}\n{traceback.format_exc()}")
        return pd.DataFrame(), "error"
    
def get_paginated_products(all_products_df, page=0, products_per_page=5):
    """Helper function to get a specific page of products from the full results."""
    if all_products_df.empty:
        logging.info("No products available for pagination")
        return pd.DataFrame(columns=["product_id", "product", "description", "price", "size", "color", "stock", "link", "photo", "relevance"]), False
    
    start_idx = page * products_per_page
    end_idx = start_idx + products_per_page

    paginated_products = all_products_df.iloc[start_idx:end_idx]

    has_more = end_idx < len(all_products_df)
    
    logging.info(f"Pagination: Page {page}, showing products {start_idx+1}-{min(end_idx, len(all_products_df))} of {len(all_products_df)}")
    logging.info(f"Has more pages: {has_more}")
    
    return paginated_products, has_more

def detect_more_products_request(user_input: str) -> bool:
    """Detect if user is asking for more products"""
    more_patterns = [
        r'\b(more|other|another|additional|different|else)\s+(product|item|option|choice|recommendation)',
        r'\b(show|give|find|get)\s+(me\s+)?(more|other|another|additional)',
        r'\b(what|anything)\s+else',
        r'\b(more|other)\s+(suggestion|option|choice)',
        r'\belse\s+(do\s+you\s+have|available)',
        r'\b(lain|lainnya|yang lain|lagi)\b',
        r'\b(tunjukkan|carikan|kasih|coba)\s+(yang\s+)?(lain|lainnya)',
        r'\b(ada\s+)?(yang\s+)?(lain|lainnya)',
        r'\b(produk|barang|item)\s+(lain|lainnya)',
        r'\b(pilihan|opsi)\s+(lain|lainnya)',
        r'\b(apa\s+lagi|apalagi)',
        r'\b(selain\s+itu|besides)',
        r'\b(lebih\s+banyak|more)',
        r'\bapa\s+lagi\b',
        r'\blainnya\b.*\b(produk|barang|pilihan)',
    ]
    
    user_input_lower = user_input.lower().strip()
    
    simple_responses = ["yes", "ya", "iya", "ok", "okay", "sure", "tentu", "no", "tidak", "nope", "ga", "engga"]
    if user_input_lower in simple_responses:
        return False
    
    if len(user_input_lower.split()) <= 2 and user_input_lower not in ["apa lagi", "yang lain", "show more"]:
        return False
    
    for pattern in more_patterns:
        if re.search(pattern, user_input_lower):
            logging.info(f"Detected specific 'more products' request: {user_input}")
            return True
    
    return False

def extract_budget_from_text(text):
    """Extract budget information with physical context detection."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    explicit_budget_keywords = ['budget', 'anggaran', 'harga', 'price', 'biaya', 'cost']
    constraint_indicators = [
        'dibawah', 'under', 'maksimal', 'max', 'kurang dari', 'less than',
        'diatas', 'over', 'minimal', 'min', 'lebih dari', 'more than',
        'sekitar', 'around', 'kisaran', 'range'
    ]
    
    currency_patterns = [
        r'\b(\d+)rb\b', r'\b(\d+)ribu', r'\b(\d+)k\b', r'\b(\d+)jt\b', r'\b(\d+)juta',
        r'\brp\.?\s*\d+\b', r'\bidr\b'
    ]
    
    has_explicit_budget = any(keyword in text_lower for keyword in explicit_budget_keywords)
    has_constraint = any(indicator in text_lower for indicator in constraint_indicators)
    has_currency = any(re.search(pattern, text_lower) for pattern in currency_patterns)
    
    if not (has_explicit_budget or (has_constraint and has_currency) or has_currency):
        return None
    
    def convert_to_rupiah(amount_str, unit):
        try:
            amount = int(amount_str)
            if unit in ['rb', 'ribu', 'k']: return amount * 1000
            elif unit in ['jt', 'juta']: return amount * 1000000
            elif unit == '000': return amount * 1000
            else: return amount
        except:
            return None
    
    budget_patterns = [
        (r'(?:budget|anggaran|harga)?\s*(?:antara|between)?\s*(\d+)(?:rb|ribu|k|jt|juta)?\s*(?:-|sampai|hingga|to)\s*(\d+)(?:rb|ribu|k|jt|juta)?', "RANGE"),
        (r'(?:dibawah|under|maksimal|max|kurang\s+dari|less\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "MAX"),
        (r'(?:diatas|over|minimal|min|lebih\s+dari|more\s+than)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "MIN"),
        (r'(?:budget|anggaran|sekitar|around|kisaran)\s*(?:rp\.?\s*)?(\d+)(?:rb|ribu|k|jt|juta|000)?', "EXACT"),
        (r'\b(\d+)(?:rb|ribu|k|jt|juta)\b', "STANDALONE"),
    ]
    
    for pattern, pattern_type in budget_patterns:
        matches = list(re.finditer(pattern, text_lower))
        
        for match in matches:
            groups = match.groups()
            match_text = match.group(0)
            
            if pattern_type == "RANGE" and len(groups) >= 2 and groups[0] and groups[1]:
                unit = 'rb' if any(x in match_text for x in ['rb', 'ribu', 'k']) else 'jt' if 'jt' in match_text else None
                min_price = convert_to_rupiah(groups[0], unit)
                max_price = convert_to_rupiah(groups[1], unit)
                if min_price and max_price: return (min(min_price, max_price), max(min_price, max_price))
            
            elif len(groups) >= 1 and groups[0]:
                unit = None
                if any(x in match_text for x in ['rb', 'ribu', 'k']): unit = 'rb'
                elif any(x in match_text for x in ['jt', 'juta']): unit = 'jt'
                elif '000' in match_text: unit = '000'
                
                amount = convert_to_rupiah(groups[0], unit)
                if amount:
                    if pattern_type == "MAX": return (None, amount)
                    elif pattern_type == "MIN": return (amount, None)
                    elif pattern_type in ["EXACT", "STANDALONE"]:
                        min_range = int(amount * 0.8)
                        max_range = int(amount * 1.2)
                        return (min_range, max_range)
    
    return None

def detect_and_update_gender(user_input: str, user_context: Dict, force_update: bool = False) -> Optional[str]:
    """
    Detect gender from user input and update context, prioritizing new explicit mentions.
    If no new gender is detected, it defaults to the existing gender in user_context.
    """
    user_input_lower = user_input.lower()
    detected_gender_in_input = None
    detected_term_in_input = None
    confidence_in_input = 0

    gender_patterns = {
        'male': [r'\b(' + '|'.join([term for term in fashion_categories.GENDER_TERMS if term in ['pria', 'laki-laki', 'male', 'man', 'cowok', 'cowo']]) + r')\b'],
        'female': [r'\b(' + '|'.join([term for term in fashion_categories.GENDER_TERMS if term in ['perempuan', 'wanita', 'female', 'woman', 'cewek', 'cewe']]) + r')\b']
    }
    
    for gender_category, patterns in gender_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, user_input_lower)
            if match:
                detected_gender_in_input = gender_category
                detected_term_in_input = match.group(1) if match.lastindex else match.group(0)
                confidence_in_input = 10.0
                break
        if detected_gender_in_input: break
            
    current_gender_in_context = user_context.get("user_gender", {})
    existing_category = current_gender_in_context.get("category")

    if detected_gender_in_input:
        user_context["user_gender"] = {"category": detected_gender_in_input, "term": detected_term_in_input, "confidence": confidence_in_input, "last_updated": datetime.now().isoformat()}
        print(f"Gender detected and saved from current input: {detected_gender_in_input} (term: {detected_term_in_input}, confidence: {confidence_in_input})")
        return detected_gender_in_input
    elif existing_category and not force_update:
        print(f"No new gender detected, using existing: {existing_category} (confidence: {current_gender_in_context.get('confidence', 0):.1f})")
        return existing_category
    else:
        print("No gender detected from current input and no prior gender stored.")
        user_context["user_gender"] = {"category": None, "term": None, "confidence": 0, "last_updated": datetime.now().isoformat()}
        return None

# ================================
# PRODUCT SEARCH FLOW
# ================================

async def enhanced_product_search_flow(user_input: str, user_context: Dict, db: AsyncSession, user_language: str, session_id: str, is_multi_item_request_from_caller: bool):
    is_multi_item_for_search = is_multi_item_request_from_caller

    print(f"DEBUG: enhanced_product_search_flow entered. is_multi_item_active (from user_context) = {user_context.get('is_multi_item_active', 'NOT_FOUND')}. is_multi_item_request_from_caller = {is_multi_item_request_from_caller}")

    if is_multi_item_for_search: 
        print(f"DEBUG: enhanced_product_search_flow branching to handle_multi_item_search. is_multi_item_for_search = {is_multi_item_for_search}")
        return await handle_multi_item_search(user_context, db, is_multi_item_for_search)
    else:
        print(f"DEBUG: enhanced_product_search_flow branching to handle_single_item_search. is_multi_item_for_search = {is_multi_item_for_search}")
        return await handle_single_item_search(user_context, db, is_multi_item_for_search)
    
async def handle_multi_item_search(user_context, db, is_multi_item):
    """For multi-item: get keywords for each clothing category separately"""
    linked_system = user_context.get('linked_keyword_system')
    if not linked_system: return pd.DataFrame(), "no_keywords"
    
    all_results = []
    
    for clothing_category, chain in linked_system.chains.items():
        print(f"\nSearching for category: {clothing_category}")
        category_keywords = chain.get_all_keywords()
        if not category_keywords: continue
            
        print(f"   Keywords for {clothing_category}: {category_keywords[:5]}")
        
        products_df, search_status = await fetch_products_from_db(
            db=db,
            top_keywords=category_keywords,
            max_results=5,
            gender_category=user_context.get("user_gender", {}).get("category"),
            budget_range=user_context.get("budget_range"),
            focus_category=clothing_category,
            is_multi_item_request=is_multi_item
        )
        
        if not products_df.empty:
            products_df['source_category'] = clothing_category
            all_results.append(products_df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        products_per_page = user_context["product_cache"].get("products_per_page", 5)
        balanced_results = balance_multi_item_results(combined_df, products_per_page=products_per_page)
        return balanced_results, "multi_item_success"
    else: return pd.DataFrame(), "no_products_found"

async def handle_single_item_search(user_context, db, is_multi_item):
    """For single-item: use flattened keywords from all chains"""
    all_keywords = get_keywords_for_product_search(user_context)
    if not all_keywords:
        return pd.DataFrame(), "no_keywords"

    print(f"\nSingle-item search with {len(all_keywords)} keywords")
    print(f"   Top keywords: {all_keywords[:5]}")

    results_df, search_status = await fetch_products_from_db(
        db=db,
        top_keywords=all_keywords,
        max_results=15,
        gender_category=user_context.get("user_gender", {}).get("category"),
        budget_range=user_context.get("budget_range"),
        is_multi_item_request=is_multi_item
    )

    return results_df, search_status

def balance_multi_item_results(combined_df, max_total=15, products_per_page=5):
    if combined_df.empty: return combined_df
    
    categories = combined_df['source_category'].unique()
    top_products_by_category = {}
    for category in categories:
        category_df = combined_df[combined_df['source_category'] == category]
        top_products_by_category[category] = category_df.sort_values(by='relevance', ascending=False).to_dict(orient='records')
        
    balanced_results = []
    current_indices = {cat: 0 for cat in categories}
    total_added = 0
    
    for category in categories:
        if current_indices[category] < len(top_products_by_category[category]):
            balanced_results.append(top_products_by_category[category][current_indices[category]])
            current_indices[category] += 1
            total_added += 1
            if total_added >= products_per_page: break
    
    added_product_ids = {item['product_id'] for item in balanced_results}

    while total_added < max_total:
        added_in_round = 0
        for category in categories:
            if current_indices[category] < len(top_products_by_category[category]):
                next_product = top_products_by_category[category][current_indices[category]]
                if next_product['product_id'] not in added_product_ids:
                    balanced_results.append(next_product)
                    added_product_ids.add(next_product['product_id'])
                    current_indices[category] += 1
                    total_added += 1
                    added_in_round += 1
                    if total_added >= max_total: break
            
        if added_in_round == 0: break

    final_df = pd.DataFrame(balanced_results)
    if not final_df.empty: final_df = final_df.sort_values(by='relevance', ascending=False).reset_index(drop=True)
    
    return final_df.head(max_total)

# ================================
# FASHION SEMANTIC SYSTEM NEW CLASSES
# ================================

class FashionSemanticSystem:
    def __init__(self):
        self.model = None
        self.product_embeddings = {}
        self.embedding_cache = {}
        self.cultural_context = self._build_indonesian_fashion_context()
        self.fashion_vocabulary = self._build_fashion_vocabulary()
        self.initialize_model()
    
    def initialize_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
                print("Loaded multilingual sentence transformer for Indonesian support")
                return
            except Exception as e:
                print(f"Failed to load multilingual model: {e}")
                try:
                    self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                    print("Loaded English sentence transformer")
                    return
                except Exception as e:
                    print(f"Failed to load sentence transformer: {e}")
        
        print("Using fallback TF-IDF approach for semantic similarity (sentence-transformers not available)")
        self.model = None
    
    def _build_indonesian_fashion_context(self):
        return {'traditional_wear': {'batik': ['traditional', 'cultural', 'patterned', 'formal', 'Indonesian'], 'kebaya': ['traditional', 'formal', 'elegant', 'cultural', 'Indonesian'], 'sarong': ['traditional', 'casual', 'comfortable', 'wrap']}, 'climate_appropriate': {'tropical': ['lightweight', 'breathable', 'cotton', 'linen', 'airy'], 'humid': ['moisture-wicking', 'quick-dry', 'loose-fit', 'ventilated'], 'hot_weather': ['light colors', 'short sleeves', 'sun protection', 'UV resistant']}, 'body_types': {'petite': ['asian fit', 'small frame', 'proportional', 'fitted'], 'average': ['standard fit', 'regular', 'balanced'], 'curvy': ['flattering cut', 'comfortable fit', 'accentuating']}, 'local_preferences': {'modest': ['covered', 'conservative', 'appropriate', 'respectful'], 'hijab_friendly': ['loose fit', 'long sleeves', 'high neck', 'modest'], 'work_appropriate': ['professional', 'modest', 'formal', 'office-suitable']}}
    
    def _build_fashion_vocabulary(self):
        return {'kaos': ['shirt', 't-shirt', 'top', 'tee', 'casual shirt'], 'kemeja': ['shirt', 'blouse', 'dress shirt', 'button-up', 'formal shirt'], 'celana': ['pants', 'trousers', 'bottoms', 'slacks'], 'jaket': ['jacket', 'blazer', 'coat', 'outerwear'], 'gaun': ['dress', 'gown', 'frock'], 'rok': ['skirt', 'bottom'], 'sepatu': ['shoes', 'footwear', 'sneakers', 'boots'], 'kasual': ['casual', 'relaxed', 'informal', 'everyday'], 'formal': ['formal', 'business', 'professional', 'dressy'], 'santai': ['comfortable', 'leisure', 'relaxed', 'easy-going'], 'elegan': ['elegant', 'sophisticated', 'classy', 'refined'], 'tradisional': ['traditional', 'cultural', 'ethnic', 'heritage'], 'modern': ['modern', 'contemporary', 'current', 'trendy'], 'longgar': ['loose', 'oversized', 'relaxed fit', 'baggy'], 'ketat': ['tight', 'fitted', 'slim', 'body-hugging'], 'pas': ['perfect fit', 'just right', 'well-fitted'], 'hitam': ['black', 'dark'], 'putih': ['white', 'cream', 'ivory'], 'merah': ['red', 'crimson', 'scarlet'], 'biru': ['blue', 'navy', 'azure'], 'hijau': ['green', 'emerald', 'olive'], 'kuning': ['yellow', 'gold', 'amber']}
    
    def preprocess_fashion_text(self, text: str, user_profile: Optional[Dict] = None) -> str:
        if not text: return ""
        enhanced_text = text.lower().strip()
        main_translations = {'blazer': 'blazer jas', 'jas': 'blazer jas', 'kemeja': 'kemeja shirt', 'shirt': 'kemeja shirt', 'celana': 'celana pants', 'pants': 'celana pants', 'rok': 'rok skirt', 'skirt': 'rok skirt', 'dress': 'dress gaun', 'gaun': 'dress gaun', 'jaket': 'jaket jacket', 'jacket': 'jaket jacket'}
        modified = False
        for indonesian, translation in main_translations.items():
            if indonesian in enhanced_text.split():
                enhanced_text = enhanced_text.replace(indonesian, translation)
                modified = True
        if not modified and len(enhanced_text.split()) < 3:
            for indonesian, translations in self.fashion_vocabulary.items():
                if indonesian in enhanced_text:
                    enhanced_text = f"{enhanced_text} {' '.join(translations)}"
                    break
        print(f"CLEAN PREPROCESSING: '{text}' → '{enhanced_text}'")
        return enhanced_text

    def get_semantic_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        if use_cache and text in self.embedding_cache: return self.embedding_cache[text]
        processed_text = self.preprocess_fashion_text(text)
        embedding = np.array([])
        if self.model and SENTENCE_TRANSFORMERS_AVAILABLE:
            if processed_text: embedding = self.model.encode([processed_text], convert_to_tensor=False)[0]
            else: embedding = np.zeros(self.model.get_sentence_embedding_dimension())
        else:
            if tfidf_vectorizer is not None and processed_text: embedding = tfidf_vectorizer.transform([processed_text]).toarray()[0]
            else: embedding = np.zeros(100)
        if use_cache: self.embedding_cache[text] = embedding
        return embedding
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_semantic_embedding(text1)
        emb2 = self.get_semantic_embedding(text2)
        if emb1.shape != emb2.shape or np.linalg.norm(emb1) == 0 or np.linalg.norm(emb2) == 0: return 0.0
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

semantic_system = FashionSemanticSystem()

class EnhancedSemanticProductMatcher:
    def __init__(self, semantic_system: FashionSemanticSystem):
        self.semantic_system = semantic_system
        self.product_embeddings = {}
        self.products_df = None
        self.embeddings_cache_dir = "embeddings_cache"
        self.embeddings_file = os.path.join(self.embeddings_cache_dir, "product_embeddings.pkl")
        self.products_file = os.path.join(self.embeddings_cache_dir, "products_data.pkl")
        self.metadata_file = os.path.join(self.embeddings_cache_dir, "metadata.pkl")
        os.makedirs(self.embeddings_cache_dir, exist_ok=True)

    def clean_semantic_query(self, query):
        cleaned = re.sub(r'\b\d+\b|\b\w\b', '', query)
        noise_words = ['carikan', 'tunjukkan', 'show', 'find', 'cari', 'aja', 'saja', 'only', 'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'for', 'to', 'with', 'on', 'at', 'by', 'from', 'about', 'as', 'in', 'of', 'off', 'out', 'up', 'down', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now', 'm', 're', 've', 'll', 'd', 't', 's', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'than', 'through', 'this', 'that', 'these', 'those']
        words = cleaned.split()
        filtered_words = []
        for word in words:
            if word.lower() not in noise_words and len(word) > 1: filtered_words.append(word)
        result = ' '.join(filtered_words[:5])
        return result.strip()

    def matches_main_category(self, query, product):
        query_lower = query.lower()
        product_name_lower = product['product_name'].lower()
        product_detail_lower = product['product_detail'].lower()
        
        category_rules = {'blazer': ['blazer', 'jas'], 'jas': ['blazer', 'jas'], 'shirt': ['shirt', 'kemeja', 'blouse', 'atasan', 'kaos', 't-shirt'], 'kemeja': ['shirt', 'kemeja', 'blouse', 'atasan', 'kaos', 't-shirt'], 'pants': ['pants', 'celana', 'trousers', 'jeans', 'leggings'], 'celana': ['pants', 'celana', 'trousers', 'jeans', 'leggings'], 'skirt': ['skirt', 'rok'], 'rok': ['skirt', 'rok'], 'dress': ['dress', 'gaun', 'terusan'], 'gaun': ['dress', 'gaun', 'terusan'], 'jacket': ['jaket', 'jacket', 'coat', 'mantel'], 'jaket': ['jaket', 'jacket', 'coat', 'mantel'], 'shorts': ['shorts', 'celana pendek']}
        
        query_clothing_terms = []
        for main_term, valid_terms in category_rules.items():
            if any(q_word in query_lower for q_word in [main_term] + self.semantic_system.fashion_vocabulary.get(main_term, [])):
                query_clothing_terms.extend(valid_terms)
        
        if not query_clothing_terms: return True
        if any(term in product_name_lower or term in product_detail_lower for term in query_clothing_terms): return True
        
        return False
    
    async def preprocess_products(self, db: AsyncSession, force_refresh=False):
        
        try:
            variant_subquery = (select(ProductVariant.product_id, func.min(ProductVariant.product_price).label('min_price'), func.group_concat(ProductVariant.size.distinct()).label('available_sizes'), func.group_concat(ProductVariant.color.distinct()).label('available_colors'), func.sum(ProductVariant.stock).label('total_stock')).where(ProductVariant.stock > 0).group_by(ProductVariant.product_id).subquery())
            query = (select(Product.product_id, Product.product_name, Product.product_detail, Product.product_seourl, Product.product_gender, variant_subquery.c.min_price, variant_subquery.c.available_sizes, variant_subquery.c.available_colors, variant_subquery.c.total_stock, ProductPhoto.productphoto_path).select_from(Product).join(variant_subquery, Product.product_id == variant_subquery.c.product_id).join(ProductPhoto, Product.product_id == ProductPhoto.product_id).where(variant_subquery.c.total_stock > 0))
            
            result = await db.execute(query)
            all_products_raw = result.fetchall()
            
            if not all_products_raw:
                print("No products found for preprocessing")
                self.products_df = pd.DataFrame()
                self.product_embeddings = {}
                return
            
            print(f"Found {len(all_products_raw)} products in database")

            current_db_product_ids = {row[0] for row in all_products_raw}

            if not force_refresh and os.path.exists(self.metadata_file) and os.path.exists(self.products_file) and os.path.exists(self.embeddings_file):
                try:
                    import pickle
                    with open(self.metadata_file, 'rb') as f: metadata = pickle.load(f)
                    cached_timestamp = metadata.get('timestamp')
                    cached_product_ids = metadata.get('product_ids_checksum')
                    
                    if cached_product_ids == current_db_product_ids and cached_timestamp and (datetime.now() - datetime.fromisoformat(cached_timestamp)).days < 7:
                        with open(self.products_file, 'rb') as f: self.products_df = pickle.load(f)
                        with open(self.embeddings_file, 'rb') as f: self.product_embeddings = pickle.load(f)
                        print("Loaded product data and embeddings from cache.")
                        return
                    else: print("Cache outdated or product data changed. Re-generating embeddings.")
                except Exception as e:
                    print(f"Error loading cache: {e}. Re-generating embeddings.")
                    force_refresh = True

            product_data = []
            for product_row in all_products_raw: product_data.append({"product_id": product_row[0], "product_name": product_row[1], "product_detail": product_row[2], "price": product_row[5], "gender": product_row[4], "sizes": product_row[6], "colors": product_row[7], "stock": product_row[8], "photo": product_row[9], "seourl": product_row[3]})
            self.products_df = pd.DataFrame(product_data)
            
            print(f"Created product DataFrame with {len(self.products_df)} products for embedding generation.")
            
            new_product_embeddings = {}
            for idx, row in self.products_df.iterrows():
                product_text = f"{row['product_name']} {row['product_detail']} {row['colors']} {row['gender']} {row['sizes']}"
                new_product_embeddings[row['product_id']] = self.semantic_system.get_semantic_embedding(product_text, use_cache=True)
            self.product_embeddings = new_product_embeddings
            
            import pickle
            with open(self.products_file, 'wb') as f: pickle.dump(self.products_df, f)
            with open(self.embeddings_file, 'wb') as f: pickle.dump(self.product_embeddings, f)
            with open(self.metadata_file, 'wb') as f: pickle.dump({'timestamp': datetime.now().isoformat(), 'product_ids_checksum': current_db_product_ids}, f)
            print("Generated and saved new product embeddings to cache.")

        except Exception as e:
            logging.error(f"Error in preprocess_products: {str(e)}\n{traceback.format_exc()}")
            print(f"Error preprocessing products: {str(e)}")

enhanced_matcher = EnhancedSemanticProductMatcher(semantic_system)

# ================================
# SETUP AND INITIALIZATION
# ================================

openai = OpenAI(api_key="sk-V4UGt-FNIde85D7t0zrvZbVG5eolZDsE8awXTAuJYgT3BlbkFJToj--_okCjQAcwzYu4ZC6JDX8kznTJhzruBhL9Q5YA")

client = OpenAI()
app = FastAPI()
app.mount("/static", StaticFiles(directory="Chatbot/static"), name="static")
templates = Jinja2Templates(directory="Chatbot/templates")
UPLOAD_DIR = "static/uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "jfif"}
Base = declarative_base()
DATABASE_URL = "mysql+aiomysql://root:@localhost:3306/ecommerce"
engine = create_async_engine(DATABASE_URL, echo=True)
database = Database(DATABASE_URL)

class Product(Base):
    __tablename__ = "product"
    product_id = Column(Integer, primary_key=True, index=True)
    product_name = Column(String(255), nullable=False)
    product_detail = Column(String(255), nullable=False)
    product_price = Column(Float, nullable=False)
    product_seourl = Column(String(255), nullable=False)
    product_gender = Column(String(255), nullable=False)

class ProductPhoto(Base):  
    __tablename__ = "product_photo"
    productphoto_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, nullable=False)
    productphoto_path = Column(String(255), nullable=False)
    productphoto_order = Column(Integer, nullable=False)

class ProductVariant(Base):
    __tablename__ = "product_variant"
    variant_id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, nullable=False)
    size = Column(String(50), nullable=False)
    color = Column(String(50), nullable=False)
    stock = Column(Integer, nullable=False)
    product_price = Column(Float, nullable=False)

class ChatHistoryDB(Base):
    __tablename__ = "chat_history"
    message_id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False)
    message_type = Column(String(20), nullable=False)
    content = Column(String(2000), nullable=False)
    timestamp = Column(DateTime, server_default=func.now())

class ChatMessage(BaseModel):
    session_id: str
    message_type: str
    content: str

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessage]

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_db():
    async with AsyncSession(engine) as session:
        yield session
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

if not os.path.exists(UPLOAD_DIR): os.makedirs(UPLOAD_DIR)

@Language.factory("language_detector")
def get_language_detector(nlp, name): return LanguageDetector()

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    try: nlp = spacy.load("en_core_web_sm")
    except OSError:
        import sys
        print("Please install 'en_core_web_sm' or 'en_core_web_lg' spacy model.")
        sys.exit()

if not spacy.tokens.Doc.has_extension("language"): spacy.tokens.Doc.set_extension("language", default={})
nlp.add_pipe("language_detector", last=True)
chat_responses = []

async def is_small_talk(input_text):
    greetings = ["hello", "hi", "hey", "hi there", "hello there", "good morning", "good afternoon", "good evening", "selamat pagi", "pagi", "selamat siang", "siang", "malam", "selamat malam"]
    return input_text.lower() in greetings or re.match(r"^\s*(hi|hello|hey)\s*$", input_text, re.IGNORECASE)

class SessionLanguageManager:
    def __init__(self): self.session_languages = {}
    def detect_or_retrieve_language(self, session_id, text):
        if session_id in self.session_languages: return self.session_languages[session_id]
        try:
            if text and text.strip():
                lang = detect(text)
                self.session_languages[session_id] = lang
                return lang
            return "unknown"
        except Exception as e:
            print(f"Language detection error: {e}")
            return "unknown"
            
    def reset_session(self, session_id):
        if session_id in self.session_languages: del self.session_languages[session_id]

def detect_language(text):
    try:
        if not text or not text.strip(): raise ValueError("Input text is empty or invalid.")
        return detect(text)
    except Exception as e:
        print(f"Language detection error: {e}")
        return "unknown"

session_manager = SessionLanguageManager()
session_contexts: Dict[str, Dict[str, Any]] = {}

def translate_text(text, target_language, session_id=None):
    """
    Translates text to the target language, with a fallback to the original text on error.
    """
    if not text or not isinstance(text, str):
        return text 

    try:
        source_language = session_manager.detect_or_retrieve_language(session_id, text)
        if source_language == target_language:
            return text
        
        translated_text = MyMemoryTranslator(source=source_language, target=target_language).translate(text)
        return translated_text
    except Exception as e:
        logging.error(f"Error during translation for text '{text[:50]}...': {e}")
        return text

def render_markdown(text: str) -> str:
    extensions = ['tables', 'nl2br', 'fenced_code', 'smarty']
    html_content = markdown(text, extensions=extensions)
    return html_content

cloudinary.config(cloud_name="dn0xl1q3g", api_key="252519847388784", api_secret="pzLNZgLzfMQ9bmwiIRoyjRFqqkU")

def upload_to_cloudinary(file_location):
    try:
        if not os.path.exists(file_location):
            logging.error(f"File not found at location: {file_location}")
            return None
            
        file_size = os.path.getsize(file_location)
        if file_size == 0:
            logging.error("Uploaded file has 0 bytes")
            return None
            
        logging.info(f"Uploading file to Cloudinary: {file_location}, Size: {file_size} bytes")
        transformation = {'quality': 'auto', 'fetch_format': 'auto'}
        response = cloudinary.uploader.upload(file_location, folder="uploads/", transformation=transformation)
        logging.info(f"Cloudinary upload successful: {response['url']}")
        return response['url']
    except Exception as e:
        logging.error(f"Cloudinary upload error: {e}")
        return None

image_analysis_cache = {}

async def analyze_uploaded_image(image_url: str):
    try:
        if not image_url: return "Error: No image URL provided."
        
        logging.info(f"Analyzing image at URL: {image_url}")
        print(f"Analyzing image URL at: {image_url}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create (
                    model="gpt-4o",
                    messages=[{
                        "role": "system", 
                        "content": (
                            "You are an AI fashion consultant specializing in detailed analysis of clothing and appearance." 
                            "Provide comprehensive, structured analysis focusing on the following aspects:\n\n"
                            "1. Type & Category: Precisely identify the specific garment type(s)\n"
                            "2. Color & Pattern: Describe dominant and accent colors, pattern types\n"
                            "3. Fabric & Texture: Identify material composition if visible\n"
                            "4. Style Classification: Casual, formal, business, streetwear, etc.\n"
                            "5. Design Elements: Note distinctive features, cuts, shapes\n"
                            "6. Fit Profile: Loose, tight, oversized, tailored\n\n"
                            "FOR PEOPLE IN IMAGES:\n"
                            "1. Body Structure: Height range, build type (slender, athletic, etc.)\n"
                            "2. Proportions: Shoulder width, waist-to-hip ratio, limb length\n"
                            "3. Physical Features: Skin tone (using neutral descriptors)\n\n"
                            "FORMAT YOUR RESPONSE IN THESE DISTINCT SECTIONS:\n"
                            "CLOTHING ANALYSIS: [clothing details organized by category]\n"
                            "PHYSICAL ATTRIBUTES: [objective body structure information]\n"
                            "KEY STYLE ELEMENTS: [3-5 most distinctive features as simple bullet points]\n\n"
                            "Important rules:\n"
                            "- Use objective, technical language and avoid subjective assessments\n"
                            "- Never comment on attractiveness or make value judgments\n"
                            "- Avoid gender assumptions unless clearly evident\n"
                            "- Be specific and detailed with the clothing analysis\n"
                            "- Focus on details that would be relevant for finding similar items"
                            )}, 
                            {"role": "user", 
                             "content": [{
                                 "type": "text", 
                                 "text": "Analyze this image in detail, focusing on clothing items and relevant physical attributes for fashion recommendations."
                                }, 
                                {"type": "image_url", "image_url": {"url": image_url}}] 
                        }],
                    max_tokens=600,
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
                image_analysis_cache[image_url] = analysis
                return analysis
            
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.info(f"Failed to analyze image at URL: {image_url}. Error: {str(e)}\n{traceback.format_exc()}")
                    print(f"Failed to analyze image at URL: {e.args}\n{traceback.format_exc()}") # Corrected print statement
                    return f"Error: Unable to analyse image. Please try again or use text description instead."
                else:
                    logging.info(f"Retrying image analysis after error: {str(e)}")
                    await asyncio.sleep(2)

    except Exception as e:
        print(f"Error during image analysis: {e}")
        return f"Error: {str(e)}"

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request): return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/")
async def upload(user_input: str = Form(None), file: UploadFile = None):
    if not file and not user_input: return JSONResponse(content={"success": False, "error": "No input or file received"})
    try:
        if file:
            file_extension = file.filename.split(".")[-1].lower()
            if file_extension not in ALLOWED_EXTENSIONS: raise HTTPException(status_code=400, detail="Invalid file type.")
            file_content = await file.read()
            file_size = len(file_content)
            logging.info(f"Received file: {file.filename}, Size: {file_size} bytes")
            if file_size == 0:
                logging.error("Uploaded file has 0 bytes")
                return JSONResponse(content={"success": False, "error": "Uploaded file is empty."})
            if file_size > 5 * 1024 * 1024: return JSONResponse(content={"success": False, "error": "File size exceeds 5MB limit."})
            if not os.path.exists(UPLOAD_DIR):
                os.makedirs(UPLOAD_DIR)
                logging.info(f"Created upload directory: {UPLOAD_DIR}")
            unique_id = uuid.uuid4()
            sanitized_filename = slugify(file.filename.rsplit(".", 1)[0], lowercase=False)
            unique_filename = f"{unique_id}_{sanitized_filename}.{file_extension}"
            file_location = os.path.join(UPLOAD_DIR, unique_filename)
            with open(file_location, "wb") as file_object: file_object.write(file_content)
            if os.path.exists(file_location) and os.path.getsize(file_location) > 0: logging.info(f"File successfully saved: {file_location}, Size: {os.path.getsize(file_location)} bytes")
            else:
                logging.error(f"File not saved correctly at {file_location}")
                return JSONResponse(content={"success": False, "error": "Failed to save file."})
            image_url = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Cloudinary upload attempt {attempt+1}/{max_retries}")
                    image_url = upload_to_cloudinary(file_location)
                    if image_url:
                        logging.info(f"Successfully uploaded to Cloudinary: {image_url}")
                        break
                    else: logging.warning(f"Cloudinary upload returned None on attempt {attempt+1}")
                except Exception as e:
                    logging.error(f"Cloudinary upload attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_retries - 1: logging.error(f"Failed to upload to Cloudinary after {max_retries} attempts: {str(e)}")
                    else:
                        logging.info(f"Sleeping for 1 second before retry {attempt+2}")
                        time.sleep(1)
            if image_url: return JSONResponse(content={"success": True, "file_url": image_url})
            else: return JSONResponse(content={"success": False, "error": "Failed to upload image to Cloudinary."})
        elif user_input: return JSONResponse(content={"success": True})
        return JSONResponse(content={"success": False, "error": "No input or file received"})
    except Exception as e:
        logging.error(f"Error in upload endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An error occurred during file upload.")

@app.post("/chat/save")
async def save_message(message: ChatMessage, db: AsyncSession = Depends(get_db)):
    try:
        logging.info(f"Saving message: {message.session_id}, {message.message_type}, {message.content}")
        new_message = ChatHistoryDB(
            session_id=message.session_id, 
            message_type=message.message_type, 
            content=message.content)
        db.add(new_message)
        await db.commit()
        return {"success": True}
    except Exception as e:
        await db.rollback()
        logging.error(f"Error saving message: {str(e)}\nTraceback: ")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str, db: AsyncSession = Depends(get_db)):
    try:
        query = select(ChatHistoryDB).where(ChatHistoryDB.session_id == session_id).order_by(ChatHistoryDB.timestamp)
        result = await db.execute(query)
        messages = result.scalars().all()
        return ChatHistoryResponse(messages=[ChatMessage (session_id=msg.session_id, message_type=msg.message_type, content=msg.content) for msg in messages])
            
    except Exception as e:
        await db.rollback()
        logging.error(f"Error getting chat history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def chat(websocket: WebSocket, db: AsyncSession = Depends(get_db)):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    await websocket.send_text(f"{session_id}|Selamat Datang! Bagaimana saya bisa membantu Anda hari ini?\n\nWelcome! How can I help you today?")

    if session_id not in session_contexts:
        session_contexts[session_id] = {
            "message_objects" : [{
                "role": "system",
                "content": (
                    "You are a fashion consultant. Your task is to provide detailed fashion recommendations "
                    "for users based on their appearance and style preferences. Respond in a friendly, natural tone "
                    "and avoid using structured JSON or code format. Instead, communicate recommendations in conversational sentences.\n\n"
                    
                    "IMPORTANT: Always ask for their gender, weight and height, body shape, skin tone, their ethnical background and use this information as a base for your recommendations.\n\n"
                    "PRIMARY CONTEXT: INDONESIA. This is the most important rule. All your recommendations must be deeply rooted in the Indonesian context. This includes:\n"
                    "Culture & Occasion: Frame your suggestions around Indonesian social life. Consider common occasions like *'untuk ke kantor' (for the office), 'untuk nongkrong santai' (for casual hangouts), 'untuk kondangan' (for weddings/formal events),* or *'untuk acara keluarga'*. Acknowledge cultural norms of modesty ('sopan') and offer options that are stylish yet appropriate. You can suggest modern outfits with a traditional touch, like using batik or tenun patterns.\n"
                    
                    "IMPORTANT: When asking for style preferences, ALWAYS format them as bullet points with examples:\n\n"
                    
                    "**Style Preferences:**\n"
                    "• **Sleeve length preference:** Please choose from sleeveless (tank tops), short sleeve (t-shirts), 3/4 sleeve (three-quarter), or long sleeve (full coverage)\n"
                    "• **Clothing length preference:** \n"
                    "  - For tops: crop top (above waist), regular length (at waist), tunic (below waist), or longline (hip length)\n"
                    "  - For bottoms: shorts (above knee), capri (mid-calf), regular (ankle length), or long/full length (floor length)\n"
                    "• **Fit preference:** Choose from oversized (loose and baggy), regular fit (standard comfort), fitted (close to body), slim fit (tailored and snug), or loose fit (relaxed but not oversized)\n"
                    "• **Daily activity level and lifestyle:** Please specify sedentary/office work (mostly sitting), moderately active (walking, light exercise), very active/athletic (sports, gym, running), or mixed activities (combination of different activity levels)\n\n"
                    
                    "CRITICAL FORMATTING REQUIREMENTS:\n"
                    "1. Always format each clothing recommendation as a bold heading with size recommendation in parentheses\n"
                    "2. Use this exact format: **[Clothing Item Name] (Disarankan Ukuran [Size])**\n"
                    "3. Follow each heading with a detailed paragraph explanation\n"
                    "4. Always include a horizontal line (---) between different clothing recommendations\n"
                    "5. Each recommendation should be in a separate paragraph block\n\n"
                    
                    "EXAMPLE FORMAT:\n"
                    "**Kemeja Flanel Oversized (Disarankan Ukuran L)**\n"
                    "Kemeja flanel oversized memberikan tampilan santai namun tetap stylish. Cocok dipadukan dengan celana jeans atau celana panjang. Dengan aktivitas yang cukup aktif, ukuran L akan memberikan kenyamanan dan gaya yang Anda inginkan.\n\n"
                    "---\n\n"
                    "**Kemeja Polo Slim Fit (Disarankan Ukuran M)**\n"
                    "Kemeja polo slim fit cocok untuk aktivitas sehari-hari yang lebih formal. Potongan yang pas akan memberikan tampilan yang rapi dan profesional.\n\n"
                    
                    "IMPORTANT: When giving recommendations, mention specific clothing items and how they would suit the user's attributes, "
                    "such as gender, height, weight, skin tone, and consider their daily activities when suggesting appropriate fits and styles.\n\n"
                    
                    "IMPORTANT: Based on the user's measurements, body type, and daily activities, provide a size recommendation (XS, S, M, L, XL, etc.) for each suggested item, explaining why that size would work best for their lifestyle and comfort needs.\n\n"
                    
                    "Give at least 3 items recommendation.\n\n"
                    
                    "IMPORTANT: If the user asks for a specific type of clothing (such as 'kemeja', 'shirt', 'dress', 'pants', etc.), "
                    "make sure your recommendations focus directly on that specific clothing type.\n\n"
                    
                    "MANDATORY FORMATTING RULES:\n"
                    "- Every clothing item must be bold with size recommendation: **[Item Name] (Disarankan Ukuran [Size])**\n"
                    "- Each description must be a separate paragraph explaining why it suits the user\n"
                    "- Use horizontal lines (---) to separate different recommendations\n"
                    "- When asking for style preferences, ALWAYS use bullet points with clear examples\n"
                    "- End with the confirmation question about showing specific products\n\n"
                    
                    "Do not mention any specific brand of clothing.\n"
                    "After each style recommendation, always ask a yes or no question: 'Would you like to see product recommendations based on these style suggestions?' or 'Do these styles align with what you're looking for? I can show you specific products if you're interested.'\n"
                    "DO NOT provide product recommendations in your initial response - only suggest styles and wait for user confirmation."
                )
            }],
            "last_ai_response": "",
            "awaiting_confirmation": False,
            "linked_keyword_system": LinkedKeywordSystem(),
            "user_gender": {"category": None, "term": None, "confidence": 0, "last_updated": None},
            "budget_range": None,
            "product_cache": {"all_results": pd.DataFrame(), "current_page": 0, "products_per_page": 5, "has_more": False},
            "is_multi_item_active": False
        }
    
    user_context = session_contexts[session_id]
    message_objects = user_context["message_objects"]

    try:
        while True:
            try:
                data = await websocket.receive_text()
                logging.info(f"Received Websocket data: {data}")

                if "|" not in data:
                    await websocket.send_text(f"{session_id}|Invalid message format.")
                    continue

                session_id, user_input = data.split("|", 1)

                db.add(ChatHistoryDB(
                    session_id=session_id, 
                    message_type="user", 
                    content=user_input))
                await db.commit()

                user_language = session_manager.detect_or_retrieve_language(session_id, user_input)
                translated_input = translate_text(user_input, "en", session_id) if user_language != "en" else user_input

                new_keywords = extract_contextual_keywords(translated_input)
                new_budget = extract_budget_from_text(user_input)

                is_filter_only_query = (new_budget is not None) and (not new_keywords) and user_context['linked_keyword_system'].chains

                if is_filter_only_query:
                    print("Filter-only query detected. Applying filter and searching immediately.")
                    user_context['budget_range'] = new_budget
                    
                    products_df, search_status = await enhanced_product_search_flow(
                        user_input, user_context, db, user_language, session_id, user_context['is_multi_item_active']
                    )
                    
                    user_context["product_cache"]["all_results"] = products_df
                    user_context["product_cache"]["current_page"] = 0
                    paginated_products, has_more = get_paginated_products(products_df, 0, 5)
                    user_context["product_cache"]["has_more"] = has_more
                    
                    if not paginated_products.empty:
                        
                        intro_text = "Here are the results with your price filter applied:\n\n"
                        response_html = translate_text(intro_text, user_language, session_id)

                        for _, row in paginated_products.iterrows():
                             response_html += (
                                f"<div class='product-card'>\n"
                                f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                f"<div class='product-info'>\n"
                                f"<h3>{row['product']}</h3>\n"
                                f"<p class='price'>IDR {row.get('price', 'N/A')}</p>\n"
                                f"<p class='description'>{row.get('description', '')}</p>\n"
                                f"<p class='available'>Available in size: {row.get('size', 'N/A')}, Color: {row.get('color', 'N/A')}</p>\n"
                                f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                "</div>\n</div>\n"
                            )
                        if has_more:
                            more_text = "\n\nWant to see more? Just ask for 'more' or 'lainnya'!"
                            response_html += translate_text(more_text, user_language, session_id)
                        
                        await websocket.send_text(f"{session_id}|{render_markdown(response_html)}")
                    else:
                        no_products_text = "I could not find any products in that price range. Would you like to try a different budget?"
                        await websocket.send_text(f"{session_id}|{translate_text(no_products_text, user_language, session_id)}")
                    
                    user_context["awaiting_confirmation"] = True 
                    continue 

                is_positive = translated_input.strip().lower() in ["yes", "ya", "iya", "sure", "tentu", "ok", "okay"]
                is_negative = translated_input.strip().lower() in ["no", "tidak", "nope", "nah", "tidak usah"]
                is_more_request = detect_more_products_request(user_input)
                is_confirmation_action = is_positive or is_negative or is_more_request
                
                is_new_primary_query = not is_confirmation_action

                if is_new_primary_query:
                    print("New primary query detected. Resetting conversation flow.")
                    user_context["awaiting_confirmation"] = False
                    user_context["product_cache"] = {"all_results": pd.DataFrame(), "current_page": 0, "products_per_page": 5, "has_more": False}

                    is_current_input_multi = detect_multi_item_request(translated_input)
                    should_clear_chains = not is_current_input_multi and user_context.get('is_multi_item_active', False)
                    if should_clear_chains:
                        print("Shift from multi-item to single-item context. Clearing keyword chains.")
                    
                    user_context['is_multi_item_active'] = is_current_input_multi

                    user_keywords = extract_contextual_keywords(translated_input)
                    update_linked_keywords_contextual(user_context, user_keywords, is_user_input=True, is_multi_item_request=user_context['is_multi_item_active'], clear_existing_chains=should_clear_chains)

                    if not user_context['is_multi_item_active'] and len(user_context['linked_keyword_system'].chains) > 1:
                        user_context['is_multi_item_active'] = True
                        print("⬆PROMOTED to multi-item based on keyword chains.")
                    
                    detect_and_update_gender(translated_input, user_context)
                    user_context["budget_range"] = extract_budget_from_text(user_input)
                    
                    prompt_for_ai = translated_input
                    url_pattern = re.compile(r'(https?://\S+\.(?:jpg|jpeg|png|gif|bmp|webp))', re.IGNORECASE)
                    image_url_match = url_pattern.search(user_input)

                    if image_url_match:
                        image_url = image_url_match.group(1)
                        text_content = user_input.replace(image_url, "").strip()
                        analysis_result = await analyze_uploaded_image(image_url)
                        if "Error:" in analysis_result:
                            await websocket.send_text(f"{session_id}|{analysis_result}")
                            continue 
                        
                        image_keywords = extract_contextual_keywords("", analysis_result)
                        update_linked_keywords_contextual(user_context, image_keywords, is_user_input=False, is_multi_item_request=user_context['is_multi_item_active'])
                        prompt_for_ai = f"Based on this image analysis: '{analysis_result}', and the user's request: '{text_content}', provide style recommendations."

                    message_objects.append({
                        "role": "user", 
                        "content": prompt_for_ai})
                    response = openai.chat.completions.create(
                        model="gpt-3.5-turbo", 
                        messages=message_objects, 
                        temperature=0.5)
                    ai_response = response.choices[0].message.content.strip()
                    user_context["last_ai_response"] = ai_response
                    message_objects.append({"role": "assistant", "content": ai_response})

                    ai_keywords = extract_contextual_keywords("", ai_response)
                    update_linked_keywords_contextual(user_context, ai_keywords, is_user_input=False, is_multi_item_request=user_context['is_multi_item_active'])
                    
                    translated_response = translate_text(ai_response, user_language, session_id)
                    
                    # FIX #2: Removed 'await' from this synchronous call.
                    db.add(ChatHistoryDB(session_id=session_id, message_type="assistant", content=ai_response))
                    await db.commit()
                    
                    await websocket.send_text(f"{session_id}|{render_markdown(translated_response)}")
                    user_context["awaiting_confirmation"] = True

                elif user_context.get("awaiting_confirmation"):
                    if is_positive:
                        print(f"User confirmed. Searching products with multi-item flag: {user_context['is_multi_item_active']}")
                        products_df, search_status = await enhanced_product_search_flow(
                            user_input, user_context, db, user_language, session_id, user_context['is_multi_item_active']
                        )
                        user_context["product_cache"]["all_results"] = products_df
                        user_context["product_cache"]["current_page"] = 0
                        
                        paginated_products, has_more = get_paginated_products(products_df, 0, 5)
                        user_context["product_cache"]["has_more"] = has_more
                        
                        if not paginated_products.empty:
                            response_html = "Great! Here are some products you might like:\n\n"
                            response_html = translate_text(response_html, user_language, session_id)
                            for _, row in paginated_products.iterrows():
                                response_html += (
                                    f"<div class='product-card'>\n"
                                    f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                    f"<div class='product-info'>\n"
                                    f"<h3>{row['product']}</h3>\n"
                                    f"<p class='price'>IDR {row.get('price', 'N/A')}</p>\n"
                                    f"<p class='description'>{row.get('description', '')}</p>\n"
                                    f"<p class='available'>Available in size: {row.get('size', 'N/A')}, Color: {row.get('color', 'N/A')}</p>\n"
                                    f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                    "</div>\n</div>\n"
                                )
                            if has_more:
                                response_html += "\n\nWant to see more? Just ask for 'more' or 'lainnya'!"
                                response_html = translate_text(response_html, user_language, session_id)
                        else:
                            response_html = "I'm sorry, I couldn't find any products that match. Would you like to try a different search?"
                            response_html = translate_text(response_html, user_language, session_id)

                        final_response = render_markdown(translate_text(response_html, user_language, session_id))
                        await websocket.send_text(f"{session_id}|{final_response}")

                    elif is_more_request:
                        print("User asked for more products.")
                        cache = user_context["product_cache"]
                        if not cache["all_results"].empty and cache.get("has_more", False):
                            next_page = cache["current_page"] + 1
                            paginated_products, has_more = get_paginated_products(cache["all_results"], next_page, 5)
                            if not paginated_products.empty:
                                cache["current_page"] = next_page
                                cache["has_more"] = has_more
                                
                                response_html = "Here are some more options:\n\n"
                                response_html = translate_text(response_html, user_language, session_id)
                                for _, row in paginated_products.iterrows():
                                     response_html += (
                                        f"<div class='product-card'>\n"
                                        f"<img src='{row['photo']}' alt='{row['product']}' class='product-image'>\n"
                                        f"<div class='product-info'>\n"
                                        f"<h3>{row['product']}</h3>\n"
                                        f"<p class='price'>IDR {row.get('price', 'N/A')}</p>\n"
                                        f"<p class='description'>{row.get('description', '')}</p>\n"
                                        f"<p class='available'>Available in size: {row.get('size', 'N/A')}, Color: {row.get('color', 'N/A')}</p>\n"
                                        f"<a href='{row['link']}' target='_blank' class='product-link'>Buy Now</a>\n"
                                        "</div>\n</div>\n"
                                    )

                                if not has_more:
                                    response_html += "\n\nThat's all the matching products I could find."
                                    response_html = translate_text(response_html, user_language, session_id)
                                
                                final_response = render_markdown(translate_text(response_html, user_language, session_id))
                                await websocket.send_text(f"{session_id}|{final_response}")
                            else:
                                cache["has_more"] = False
                                await websocket.send_text(f"{session_id}|{translate_text('That is all I have!', user_language, session_id)}")
                        else:
                            await websocket.send_text(f"{session_id}|{translate_text('There are no more products to show. You can start a new search.', user_language, session_id)}")

                    elif is_negative:
                        print("User declined.")
                        user_context["awaiting_confirmation"] = False
                        ai_response = "Alright. What other styles or fashion advice can I help you with?"
                        translated_response = translate_text(ai_response, user_language, session_id)
                        await websocket.send_text(f"{session_id}|{translated_response}")
            
            except WebSocketDisconnect:
                logging.info(f"Websocket disconnected for session {session_id}")
                if session_id in session_contexts: del session_contexts[session_id]
                session_manager.reset_session(session_id)
                break
            except Exception as e:
                logging.error(f"Error in websocket loop: {str(e)}\n{traceback.format_exc()}")
                await websocket.send_text(f"{session_id}|I am sorry, an error occurred. Please try again.")
    except Exception as e:
        logging.error(f"Websocket connection error: {str(e)}\n{traceback.format_exc()}")
        if websocket.client_state != http.client.WebSocketState.DISCONNECTED:
            await websocket.close()

# ================================
# APP LIFECYCLE EVENTS
# ================================
@app.on_event("startup")
async def startup_event():
    print("Starting up the application")
    try:
        
        print("Ensuring database tables exist")
        await create_tables()
        print("Database tables ensured.")

        print("Fetching all products from the database for model initialization")
        async with AsyncSession(engine) as session:
            variant_subquery = (
                select(
                    ProductVariant.product_id,
                    func.min(ProductVariant.product_price).label('min_price'),
                    func.group_concat(ProductVariant.size.distinct()).label('available_sizes'),
                    func.group_concat(ProductVariant.color.distinct()).label('available_colors'),
                    func.sum(ProductVariant.stock).label('total_stock')
                )
                .where(ProductVariant.stock > 0)
                .group_by(ProductVariant.product_id)
                .subquery()
            )
            
            query = (
                select(
                    Product.product_id,
                    Product.product_name,
                    Product.product_detail,
                    Product.product_seourl,
                    Product.product_gender,
                    variant_subquery.c.min_price,
                    variant_subquery.c.available_sizes,
                    variant_subquery.c.available_colors,
                    variant_subquery.c.total_stock,
                    ProductPhoto.productphoto_path
                )
                .select_from(Product)
                .join(variant_subquery, Product.product_id == variant_subquery.c.product_id)
                .join(ProductPhoto, Product.product_id == ProductPhoto.product_id)
                .where(variant_subquery.c.total_stock > 0)
            )
            result = await session.execute(query)
            all_products_for_models = result.fetchall()
            print(f"Fetched {len(all_products_for_models)} products from database for model prep.")

        print("Initializing TF-IDF model")
        tfidf_init_success = initialize_tfidf_model(all_products_for_models)
        if tfidf_init_success:
            print("TF-IDF model initialized successfully.")
        else:
            print("TF-IDF model initialization skipped or failed. Product search will rely more on semantic embeddings and direct keyword matching.")

        print("Preparing product embeddings for semantic search (loading from cache or generating)...")
        async with AsyncSession(engine) as session_for_embeddings:
            await enhanced_matcher.preprocess_products(session_for_embeddings)
        
        # Verify if embeddings were loaded/generated successfully
        if enhanced_matcher.product_embeddings:
            print(f"Product embeddings for semantic search are ready ({len(enhanced_matcher.product_embeddings)} embeddings loaded/generated).")
        else:
            print("No product embeddings were loaded or generated. Semantic search will be limited.")

    except Exception as e:
        logging.error(f"FATAL: Application startup failed: {str(e)}\n{traceback.format_exc()}")
        print(f"FATAL ERROR during application startup: {str(e)}")
        print("Application will attempt to start, but search functionality may be impaired.")

    print("Application startup routine finished.")