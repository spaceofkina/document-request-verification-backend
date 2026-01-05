const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');
const moment = require('moment');

class TemplateService {
    generateDocument(templateType, data) {
        return new Promise((resolve, reject) => {
            try {
                const doc = new PDFDocument({ size: 'A4', margin: 50 });
                const chunks = [];
                
                doc.on('data', chunk => chunks.push(chunk));
                doc.on('end', () => {
                    const pdfBuffer = Buffer.concat(chunks);
                    resolve(pdfBuffer);
                });
                
                // Add header
                this.addHeader(doc, templateType);
                
                // Add content based on template type
                switch (templateType) {
                    case 'Barangay Clearance':
                        this.generateBarangayClearance(doc, data);
                        break;
                    case 'Certificate of Residency':
                        this.generateCertificateOfResidency(doc, data);
                        break;
                    case 'Certificate of Indigency':
                        this.generateCertificateOfIndigency(doc, data);
                        break;
                    case 'Good Moral Certificate':
                        this.generateGoodMoralCertificate(doc, data);
                        break;
                    default:
                        this.generateGenericCertificate(doc, data, templateType);
                }
                
                // Add footer
                this.addFooter(doc, data);
                
                doc.end();
            } catch (error) {
                reject(error);
            }
        });
    }

    addHeader(doc, documentType) {
        // Logo (you would add your barangay logo here)
        doc.fontSize(20)
           .font('Helvetica-Bold')
           .text('BARANGAY DOCUMENT', { align: 'center' });
        
        doc.moveDown(0.5);
        doc.fontSize(16)
           .font('Helvetica')
           .text(documentType, { align: 'center' });
        
        doc.moveDown(1);
        doc.fontSize(12)
           .text(`Control No: ${Date.now()}${Math.floor(Math.random() * 1000)}`, { align: 'right' });
        
        doc.moveTo(50, doc.y)
           .lineTo(550, doc.y)
           .stroke();
        
        doc.moveDown(1);
    }

    generateBarangayClearance(doc, data) {
        const today = moment().format('MMMM DD, YYYY');
        
        doc.fontSize(12)
           .font('Helvetica')
           .text('TO WHOM IT MAY CONCERN:', { align: 'left' });
        
        doc.moveDown(1);
        
        doc.font('Helvetica')
           .text(`This is to certify that ${data.fullName}, of legal age, ${data.civilStatus || 'single'}, Filipino, is a bona fide resident of ${data.permanentAddress}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Further certified that as per records of this Barangay, ${data.fullName} has no derogatory record filed in this office and is known to be of good moral character and a law-abiding citizen.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`This certification is issued upon the request of ${data.fullName} for ${data.purpose || 'employment purposes'}.`, { align: 'justify' });
        
        doc.moveDown(2);
        
        doc.text(`Issued this ${today} at Barangay ${data.barangayName || '[Barangay Name]'}, [City/Municipality], [Province].`, { align: 'justify' });
        
        doc.moveDown(3);
        
        // Signature section
        doc.text('_________________________', { align: 'right' });
        doc.text('Barangay Captain', { align: 'right' });
        doc.text(`Barangay ${data.barangayName || '[Barangay Name]'}`, { align: 'right' });
    }

    generateCertificateOfResidency(doc, data) {
        const today = moment().format('MMMM DD, YYYY');
        
        doc.fontSize(12)
           .font('Helvetica')
           .text('CERTIFICATION OF RESIDENCY', { align: 'center', underline: true });
        
        doc.moveDown(1);
        
        doc.text('TO WHOM IT MAY CONCERN:', { align: 'left' });
        
        doc.moveDown(1);
        
        doc.text(`This is to certify that based on the records available in this office, ${data.fullName}, ${data.age || '[Age]'} years old, is a bona fide resident of ${data.permanentAddress} since ${data.residencyStart || '[Year]'}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`This certification is issued for ${data.purpose || 'whatever legal purpose it may serve'}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Given this ${today} at Barangay ${data.barangayName || '[Barangay Name]'}.`, { align: 'justify' });
        
        doc.moveDown(3);
        
        doc.text('_________________________', { align: 'right' });
        doc.text('Barangay Secretary', { align: 'right' });
        doc.text(`Barangay ${data.barangayName || '[Barangay Name]'}`, { align: 'right' });
    }

    generateCertificateOfIndigency(doc, data) {
        const today = moment().format('MMMM DD, YYYY');
        
        doc.fontSize(12)
           .font('Helvetica')
           .text('CERTIFICATE OF INDIGENCY', { align: 'center', underline: true });
        
        doc.moveDown(1);
        
        doc.text('TO WHOM IT MAY CONCERN:', { align: 'left' });
        
        doc.moveDown(1);
        
        doc.text(`This is to certify that ${data.fullName}, ${data.age || '[Age]'} years old, is a bona fide resident of ${data.permanentAddress}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Based on the assessment conducted by this office, ${data.fullName} belongs to an indigent family with an estimated monthly income of PHP ${data.monthlyIncome || '0.00'}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`This certification is issued to avail of ${data.purpose || 'government services and assistance'}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Issued this ${today} at Barangay ${data.barangayName || '[Barangay Name]'}.`, { align: 'justify' });
        
        doc.moveDown(3);
        
        doc.text('_________________________', { align: 'right' });
        doc.text('Barangay Social Welfare Officer', { align: 'right' });
        doc.text(`Barangay ${data.barangayName || '[Barangay Name]'}`, { align: 'right' });
    }

    generateGoodMoralCertificate(doc, data) {
        const today = moment().format('MMMM DD, YYYY');
        
        doc.fontSize(12)
           .font('Helvetica')
           .text('CERTIFICATE OF GOOD MORAL CHARACTER', { align: 'center', underline: true });
        
        doc.moveDown(1);
        
        doc.text('TO WHOM IT MAY CONCERN:', { align: 'left' });
        
        doc.moveDown(1);
        
        doc.text(`This is to certify that ${data.fullName}, ${data.age || '[Age]'} years old, is a resident of ${data.permanentAddress}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Based on the records of this Barangay and from the information gathered, ${data.fullName} has not been involved in any illegal activities and is known to possess good moral character in this community.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`This certification is issued for ${data.purpose || 'academic/employment purposes'} upon the request of the above-named person.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Issued this ${today} at Barangay ${data.barangayName || '[Barangay Name]'}.`, { align: 'justify' });
        
        doc.moveDown(3);
        
        doc.text('_________________________', { align: 'right' });
        doc.text('Barangay Captain', { align: 'right' });
        doc.text(`Barangay ${data.barangayName || '[Barangay Name]'}`, { align: 'right' });
    }

    generateGenericCertificate(doc, data, documentType) {
        const today = moment().format('MMMM DD, YYYY');
        
        doc.fontSize(12)
           .font('Helvetica')
           .text(documentType.toUpperCase(), { align: 'center', underline: true });
        
        doc.moveDown(1);
        
        doc.text('TO WHOM IT MAY CONCERN:', { align: 'left' });
        
        doc.moveDown(1);
        
        doc.text(`This is to certify that ${data.fullName} is a resident of ${data.permanentAddress}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`This certification is issued for ${data.purpose || 'whatever legal purpose it may serve'}.`, { align: 'justify' });
        
        doc.moveDown(1);
        
        doc.text(`Issued this ${today} at Barangay ${data.barangayName || '[Barangay Name]'}.`, { align: 'justify' });
        
        doc.moveDown(3);
        
        doc.text('_________________________', { align: 'right' });
        doc.text('Barangay Official', { align: 'right' });
        doc.text(`Barangay ${data.barangayName || '[Barangay Name]'}`, { align: 'right' });
    }

    addFooter(doc, data) {
        const pageHeight = doc.page.height;
        const footerY = pageHeight - 100;
        
        doc.y = footerY;
        
        doc.moveTo(50, doc.y)
           .lineTo(550, doc.y)
           .stroke();
        
        doc.moveDown(0.5);
        
        doc.fontSize(8)
           .font('Helvetica-Oblique')
           .text('Note: This document is computer-generated and requires the official seal and signature of the Barangay Captain.', { align: 'center' });
        
        doc.moveDown(0.5);
        
        doc.text(`Request ID: ${data.requestId || 'N/A'} | Issued on: ${moment().format('YYYY-MM-DD HH:mm:ss')}`, { align: 'center' });
    }
}

module.exports = new TemplateService();