const Tesseract = require('tesseract.js');
const sharp = require('sharp');

class OCRService {
    constructor() {
        this.languages = ['eng', 'fil']; // English and Filipino
    }

    async extractTextFromImage(imageBuffer) {
        try {
            // Preprocess image for better OCR
            const processedBuffer = await sharp(imageBuffer)
                .greyscale()
                .normalize()
                .sharpen()
                .toBuffer();

            const result = await Tesseract.recognize(
                processedBuffer,
                this.languages,
                {
                    logger: m => console.log(m)
                }
            );

            return {
                text: result.data.text,
                confidence: result.data.confidence,
                words: result.data.words,
                lines: result.data.lines,
                paragraphs: result.data.paragraphs
            };
        } catch (error) {
            console.error('Philippine Document OCR Error:', error);
            throw error;
        }
    }

    async extractTextFromPDF(pdfBuffer) {
        try {
            const imageBuffer = await sharp(pdfBuffer, { page: 0 })
                .png()
                .toBuffer();
            
            return await this.extractTextFromImage(imageBuffer);
        } catch (error) {
            console.error('Philippine PDF OCR Error:', error);
            throw error;
        }
    }

    async extractFieldsFromID(idType, ocrResult) {
        try {
            console.log(`🔍 Extracting fields from ${idType}...`);
            
            const fields = {};
            const text = ocrResult.text?.toLowerCase() || '';
            
            // Always extract common Philippine fields
            const philippineName = this.extractPhilippineName(text);
            const philippineDate = this.extractPhilippineDate(text);
            const philippineAddress = this.extractPhilippineAddress(text);
            
            if (philippineName) fields.fullName = philippineName;
            if (philippineDate) fields.birthDate = philippineDate;
            if (philippineAddress) fields.address = philippineAddress;
            
            // FIXED: Handle idType properly
            if (idType && typeof idType === 'string') {
                // Philippine document specific patterns
                switch (idType) {
                    case 'Philippine Passport':
                        fields.passportNumber = this.extractPattern(text, /\b[a-z][0-9]{7}\b/i);
                        fields.nationality = this.extractPattern(text, /nationality[:\s]+([a-z\s]+)/i);
                        fields.placeOfBirth = this.extractPattern(text, /place of birth[:\s]+([a-z\s,]+)/i);
                        fields.idType = 'Passport';
                        break;
                        
                    case 'UMID (Unified Multi-Purpose ID)':
                        fields.umidNumber = this.extractPattern(text, /\b[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}\b/);
                        fields.crn = this.extractPattern(text, /crn[:\s]+([a-z0-9]+)/i);
                        fields.idType = 'UMID';
                        break;
                        
                    case 'Drivers License (LTO)':
                        fields.licenseNumber = this.extractPattern(text, /\b[a-z][0-9]{2}[- ]?[0-9]{2}[- ]?[0-9]{6}\b/i);
                        fields.expiryDate = this.extractPattern(text, /expir[:\s]+([0-9\/-]+)/i);
                        fields.restrictions = this.extractPattern(text, /restrictions?[:\s]+([a-z0-9,\s]+)/i);
                        fields.idType = 'Drivers License';
                        break;
                        
                    case 'National ID (PhilSys)':
                        fields.psn = this.extractPattern(text, /\b[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}[- ]?[0-9]{4}\b/);
                        fields.bloodType = this.extractPattern(text, /blood type[:\s]+([a-b][+-])/i);
                        fields.idType = 'National ID';
                        break;
                        
                    case 'SSS ID (Social Security System)':
                        fields.sssNumber = this.extractPattern(text, /\b[0-9]{2}[- ]?[0-9]{7}[- ]?[0-9]{1}\b/);
                        fields.idType = 'SSS ID';
                        break;
                        
                    case 'GSIS ID (Government Service Insurance System)':
                        fields.gsisNumber = this.extractPattern(text, /\b[0-9]{13}\b/);
                        fields.idType = 'GSIS ID';
                        break;
                        
                    case 'Voters ID':
                        fields.votersNumber = this.extractPattern(text, /\b[0-9]{3}[- ]?[0-9]{3}[- ]?[0-9]{3}[- ]?[0-9]{3}\b/);
                        fields.precinctNumber = this.extractPattern(text, /precinct[:\s]+([0-9a-z]+)/i);
                        fields.idType = 'Voters ID';
                        break;
                        
                    case 'PhilHealth ID':
                        fields.philhealthNumber = this.extractPattern(text, /\b[0-9]{2}[- ]?[0-9]{9}[- ]?[0-9]{1}\b/);
                        fields.pin = this.extractPattern(text, /pin[:\s]+([0-9a-z]+)/i);
                        fields.idType = 'PhilHealth ID';
                        break;
                        
                    case 'TIN ID (Tax Identification Number)':
                        fields.tinNumber = this.extractPattern(text, /\b[0-9]{3}[- ]?[0-9]{3}[- ]?[0-9]{3}[- ]?[0-9]{3}\b/);
                        fields.birRdo = this.extractPattern(text, /rdo[:\s]+([0-9]+)/i);
                        fields.idType = 'TIN ID';
                        break;
                        
                    case 'Postal ID':
                        fields.postalIdNumber = this.extractPattern(text, /\bpid[0-9]{9}\b/i);
                        fields.issuedDate = this.extractPattern(text, /issued[:\s]+([0-9\/-]+)/i);
                        fields.idType = 'Postal ID';
                        break;
                        
                    case 'Student ID':
                        fields.studentNumber = this.extractPattern(text, /\b[0-9]{4}[- ]?[0-9]{4}\b/);
                        fields.school = this.extractPattern(text, /school[:\s]+([a-z\s.]+)/i);
                        fields.course = this.extractPattern(text, /course[:\s]+([a-z\s]+)/i);
                        fields.idType = 'Student ID';
                        break;
                        
                    case 'Barangay ID':
                        fields.barangayIdNumber = this.extractPattern(text, /\bbid[0-9]{6}\b/i);
                        fields.barangay = this.extractPattern(text, /barangay[:\s]+([a-z\s]+)/i);
                        fields.municipality = this.extractPattern(text, /municipality[:\s]+([a-z\s]+)/i);
                        fields.idType = 'Barangay ID';
                        break;
                        
                    case 'Municipal ID':
                        fields.municipalIdNumber = this.extractPattern(text, /\bmid[0-9]{8}\b/i);
                        fields.municipality = this.extractPattern(text, /municipality[:\s]+([a-z\s]+)/i);
                        fields.province = this.extractPattern(text, /province[:\s]+([a-z\s]+)/i);
                        fields.idType = 'Municipal ID';
                        break;
                        
                    case 'Certificate of Residency':
                        fields.residencyNumber = this.extractPattern(text, /\bcert[0-9]{6}\b/i);
                        fields.purpose = this.extractPattern(text, /purpose[:\s]+([a-z\s]+)/i);
                        fields.issuedBy = this.extractPattern(text, /issued by[:\s]+([a-z\s]+)/i);
                        fields.idType = 'Certificate of Residency';
                        break;
                        
                    default:
                        console.log(`Unknown document type: ${idType}`);
                        fields.idType = 'Unknown';
                        break;
                }
            }
            
            return fields;
            
        } catch (error) {
            console.error('Field extraction error:', error);
            return {
                fullName: null,
                idNumber: null,
                address: null,
                error: 'Field extraction failed',
                details: error.message
            };
        }
    }

    extractPhilippineName(text) {
        const namePatterns = [
            /name[:\s]+([a-z]+(?:\s+[a-z]+)+)/i,
            /full name[:\s]+([a-z]+(?:\s+[a-z]+)+)/i,
            /pangalan[:\s]+([a-z]+(?:\s+[a-z]+)+)/i,
            /(?:[a-z]+)\s+[a-z]+,\s+[a-z]+(?:\s+[a-z]+)?/i // Last, First Middle format
        ];
        
        for (const pattern of namePatterns) {
            const match = text.match(pattern);
            if (match) {
                // Convert to proper case for Philippine names
                const name = match[1].replace(/\b\w/g, char => char.toUpperCase());
                return name;
            }
        }
        
        return null;
    }

    extractPhilippineDate(text) {
        const datePatterns = [
            /\b\d{2}[-\/]\d{2}[-\/]\d{4}\b/, // MM/DD/YYYY
            /\b\d{4}[-\/]\d{2}[-\/]\d{2}\b/, // YYYY/MM/DD
            /\b(?:birth|dob|date of birth|kapanganakan)[:\s]+([0-9\/-]+)/i,
            /\b(?:issued|expir|date issued)[:\s]+([0-9\/-]+)/i
        ];
        
        for (const pattern of datePatterns) {
            const match = text.match(pattern);
            if (match) {
                let date = match[1] || match[0];
                // Convert to standard format
                if (date.match(/^\d{2}[-\/]\d{2}[-\/]\d{4}$/)) {
                    const [month, day, year] = date.split(/[-\/]/);
                    return `${month}/${day}/${year}`;
                }
                return date;
            }
        }
        
        return null;
    }

    extractPhilippineAddress(text) {
        const addressPatterns = [
            /address[:\s]+([a-z0-9\s,.-]+)/i,
            /tirahan[:\s]+([a-z0-9\s,.-]+)/i,
            /(?:brgy|barangay)[:\s]+([a-z\s]+)/i,
            /(?:municipality|munisipyo|city|lungsod)[:\s]+([a-z\s]+)/i,
            /(?:province|probinsya)[:\s]+([a-z\s]+)/i
        ];
        
        const addressParts = {};
        
        for (const pattern of addressPatterns) {
            const match = text.match(pattern);
            if (match) {
                const label = pattern.source.match(/(brgy|barangay|municipality|city|province)/i);
                if (label) {
                    addressParts[label[1].toLowerCase()] = match[1].trim();
                } else {
                    addressParts.full = match[1].trim();
                }
            }
        }
        
        if (Object.keys(addressParts).length > 0) {
            return addressParts;
        }
        
        return null;
    }

    extractPattern(text, pattern) {
        const match = text.match(pattern);
        return match ? match[0] : null;
    }
}

module.exports = new OCRService();